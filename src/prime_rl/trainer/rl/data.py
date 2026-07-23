import gzip
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO, TypedDict

import msgspec
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from prime_rl.configs.trainer import FakeDataLoaderConfig
from prime_rl.trainer.batch import prepare_batch
from prime_rl.trainer.rl.packer import BasePacker, setup_packer
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.world import get_world
from prime_rl.transport import (
    MicroBatch,
    MicroBatchReceiver,
    TrainingSample,
    TransportConfig,
    setup_micro_batch_receiver,
)


class TensorMicroBatch(TypedDict):
    """A micro batch of data for training."""

    # Token level
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    advantages: Float[Tensor, "batch seq"]
    inference_logprobs: Float[Tensor, "batch seq"]
    ref_logprobs: Float[Tensor, "batch seq"] | None
    loss_mask: Bool[Tensor, "batch seq"]
    temperatures: Float[Tensor, "batch seq"]  # Per-token temperatures
    env_names: list[str]
    sequence_lengths: list[int]

    # Batch level
    lora_num_tokens: Int[Tensor, "n_loras"]
    seq_lens: Int[Tensor, "segments"]

    # MoE router replay
    routed_experts: Int[Tensor, "batch seq layers topk"] | None

    # Generic multimodal kwargs — flat dict matching the model's forward
    # signature (e.g. ``{"pixel_values": ..., "image_grid_thw": ...}`` for
    # Qwen3-VL; ``{"pixel_values": ...}`` for Gemma3-VL). The trainer
    # ``**`` -unpacks this into the forward call, so any HF VLM whose
    # processor and forward agree on kwarg names works out of the box.
    mm_kwargs: dict[str, Tensor] | None
    # mm_token_type_ids: token type per token [batch seq], int64 (0=text, 1=image, 2=video)
    mm_token_type_ids: Int[Tensor, "batch seq"] | None

    # Per-token component weight streams. ``None`` means absent: no ce/ref_kl
    # component, rl weight 1.0 on every loss-masked token.
    rl_weights: Float[Tensor, "batch seq"] | None
    ce_weights: Float[Tensor, "batch seq"] | None
    ref_kl_weights: Float[Tensor, "batch seq"] | None

    # Packer-derived metadata used for run-local debug exports.
    run_id: str | None
    run_step: int | None


class FakeDataLoader:
    def __init__(self, config: FakeDataLoaderConfig, seq_len: int, dp_world_size: int):
        self.world = get_world()
        self.dp_world_size = dp_world_size
        self.non_dp_world_size = self.world.world_size // self.dp_world_size
        self.dp_rank = self.world.rank // self.non_dp_world_size

        self.batch_size = config.batch_size
        self.num_micro_batches = self.batch_size // self.dp_world_size
        self.seq_len = seq_len
        self.generate_samples = config.generate_samples
        self.batch_counter = 0
        self.multi_run_manager = get_multi_run_manager()

    def wait_for_batch(self) -> None:
        return

    def get_batch(self) -> list[TensorMicroBatch]:
        if not self.generate_samples:
            get_micro_batch_fn = self._get_micro_batch
        else:
            get_micro_batch_fn = self._get_sample_micro_batch

        # This is a pretty ugly hack to ensure that all CP ranks in a data parallel group receive the same micro batch.
        micro_batches = []
        for micro_batch_idx in range(self.num_micro_batches):
            seed = self.dp_rank * 1000000 + self.batch_counter * 1000 + micro_batch_idx
            generator = torch.Generator().manual_seed(seed)
            micro_batches.append(get_micro_batch_fn(generator))

        self.batch_counter += 1
        return micro_batches

    def _get_sample_micro_batch(self, generator: torch.Generator) -> TensorMicroBatch:
        total_seq_len = 0
        input_ids = []
        position_ids = []
        sequence_lengths = []

        while total_seq_len < self.seq_len:
            # Generate reasonably long documents
            seq_len_to_generate = torch.randint(1, self.seq_len // 8, (1,), generator=generator).item()
            if seq_len_to_generate + total_seq_len > self.seq_len:
                seq_len_to_generate = self.seq_len - total_seq_len
            total_seq_len += seq_len_to_generate
            sequence_lengths.append(seq_len_to_generate)
            tmp_input_ids = torch.randint(0, 120000, (seq_len_to_generate,), generator=generator).long()
            tmp_position_ids = torch.arange(seq_len_to_generate).long()

            input_ids.append(tmp_input_ids)
            position_ids.append(tmp_position_ids)

        input_ids = torch.cat(input_ids, dim=0)
        position_ids = torch.cat(position_ids, dim=0)
        loss_mask = torch.ones(input_ids.shape[0], dtype=torch.bool)
        advantages = torch.randn(input_ids.shape[0], generator=generator)
        inference_logprobs = torch.randn(input_ids.shape[0], generator=generator)
        lora_num_tokens = torch.zeros(self.multi_run_manager.max_runs, dtype=torch.int32)
        lora_num_tokens[0] = input_ids.shape[0]

        return {
            "input_ids": input_ids.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
            "advantages": advantages.unsqueeze(0),
            "inference_logprobs": inference_logprobs.unsqueeze(0),
            "ref_logprobs": None,
            "temperatures": torch.ones(input_ids.shape[0]).unsqueeze(0),
            "env_names": ["fake"] * input_ids.shape[0],
            "sequence_lengths": sequence_lengths,
            "loss_mask": loss_mask.unsqueeze(0),
            "lora_num_tokens": lora_num_tokens,
            "seq_lens": torch.tensor(sequence_lengths, dtype=torch.long),
            "routed_experts": None,
            "mm_kwargs": None,
            "mm_token_type_ids": None,
            "rl_weights": None,
            "ce_weights": None,
            "ref_kl_weights": None,
            "run_id": None,
            "run_step": None,
        }

    def _get_micro_batch(self, generator: torch.Generator) -> TensorMicroBatch:
        lora_num_tokens = torch.zeros(self.multi_run_manager.max_runs, dtype=torch.int32)
        lora_num_tokens[0] = self.seq_len
        return {
            "input_ids": torch.randint(
                0,
                100,
                (
                    1,
                    self.seq_len,
                ),
                generator=generator,
            ),
            "position_ids": torch.cat([torch.arange(self.seq_len)]).unsqueeze(0),
            "advantages": torch.randn(self.seq_len, generator=generator).unsqueeze(0),
            "inference_logprobs": torch.randn(self.seq_len, generator=generator).unsqueeze(0),
            "ref_logprobs": None,
            "temperatures": torch.ones(self.seq_len).unsqueeze(0),
            "env_names": ["fake"] * self.seq_len,
            "sequence_lengths": [self.seq_len],
            "loss_mask": torch.ones(self.seq_len, dtype=torch.bool).unsqueeze(0),
            "lora_num_tokens": lora_num_tokens,
            "seq_lens": torch.tensor([self.seq_len], dtype=torch.long),
            "routed_experts": None,
            "mm_kwargs": None,
            "mm_token_type_ids": None,
            "rl_weights": None,
            "ce_weights": None,
            "ref_kl_weights": None,
            "run_id": None,
            "run_step": None,
        }


class DataLoader:
    """Loads and packs serialized training samples."""

    def __init__(
        self,
        output_dir: Path,
        start_step: int,
        dp_world_size: int,
        seq_len: int,
        pad_to_multiple_of: int,
        bin_cost: Callable[[Sequence[int]], int],
        config: TransportConfig,
        trace_path: Path | None = None,
    ):
        self.world = get_world()

        if self.world.is_master:
            if trace_path is not None:
                self.packer: BasePacker = _TracePacker(
                    trace_path=trace_path,
                    dp_world_size=dp_world_size,
                    seq_len=seq_len,
                    transport_config=config,
                    pad_to_multiple_of=pad_to_multiple_of,
                    bin_cost=bin_cost,
                    start_step=start_step,
                )
            else:
                self.packer = setup_packer(
                    dp_world_size=dp_world_size,
                    seq_len=seq_len,
                    transport_config=config,
                    pad_to_multiple_of=pad_to_multiple_of,
                    bin_cost=bin_cost,
                    start_step=start_step,
                )

        non_dp_world_size = self.world.world_size // dp_world_size
        dp_rank = self.world.rank // non_dp_world_size
        self.multi_run_manager = get_multi_run_manager()

        self.receiver: MicroBatchReceiver = setup_micro_batch_receiver(output_dir, dp_rank, start_step, config)

    def wait_for_batch(self) -> None:
        if self.world.is_master:
            self.packer._arm_watchdog()
            try:
                self.packer.pack()
            finally:
                self.packer._disarm_watchdog()
        self.receiver.wait()
        self.multi_run_manager.synchronize_state()

    def get_batch(self) -> list[TensorMicroBatch]:
        micro_batches = self.receiver.receive()
        return [self._micro_batch_to_tensor(mb) for mb in micro_batches]

    def _micro_batch_to_tensor(self, micro_batch: MicroBatch) -> TensorMicroBatch:
        """Convert a MicroBatch (msgspec struct with lists) to a TensorMicroBatch (dict with tensors)."""
        if micro_batch.lora_num_tokens is None:
            micro_batch.lora_num_tokens = [0] * self.multi_run_manager.max_runs
            micro_batch.lora_num_tokens[0] = len(micro_batch.input_ids)
        mm_kwargs: dict[str, Tensor] | None = None
        if micro_batch.mm_kwargs:
            # Each value is an EncodedTensor (dtype, shape, raw bytes).
            # No batch dim — the orchestrator concatenates per-image along
            # dim=0 generically, matching what each HF VLM's forward expects.
            mm_kwargs = {
                key: torch.frombuffer(bytearray(payload.data), dtype=_torch_dtype(payload.dtype)).reshape(payload.shape)
                for key, payload in micro_batch.mm_kwargs.items()
            }
        routed_experts = None
        packed_routed_experts = micro_batch.routed_experts
        if packed_routed_experts is not None:
            routed_experts = (
                torch.frombuffer(
                    packed_routed_experts.data,
                    dtype=_torch_dtype(packed_routed_experts.dtype),
                )
                .reshape(packed_routed_experts.shape)
                .to(torch.int32)
                .unsqueeze(0)
            )
        return TensorMicroBatch(
            input_ids=torch.tensor(micro_batch.input_ids, dtype=torch.long).unsqueeze(0),
            position_ids=torch.tensor(micro_batch.position_ids, dtype=torch.long).unsqueeze(0),
            advantages=torch.tensor(micro_batch.advantages, dtype=torch.float).unsqueeze(0),
            inference_logprobs=torch.tensor(micro_batch.inference_logprobs, dtype=torch.float).unsqueeze(0),
            ref_logprobs=torch.tensor(micro_batch.ref_logprobs, dtype=torch.float).unsqueeze(0)
            if micro_batch.ref_logprobs is not None
            else None,
            loss_mask=torch.tensor(micro_batch.loss_mask, dtype=torch.bool).unsqueeze(0),
            temperatures=torch.tensor(micro_batch.temperatures, dtype=torch.float).unsqueeze(0),
            env_names=micro_batch.env_names,
            sequence_lengths=micro_batch.sequence_lengths,
            lora_num_tokens=torch.tensor(micro_batch.lora_num_tokens, dtype=torch.int32),
            seq_lens=torch.tensor(micro_batch.seq_lens, dtype=torch.long),
            mm_kwargs=mm_kwargs,
            mm_token_type_ids=torch.tensor(micro_batch.mm_token_type_ids, dtype=torch.long).unsqueeze(0)
            if micro_batch.mm_token_type_ids is not None
            else None,
            routed_experts=routed_experts,
            rl_weights=torch.tensor(micro_batch.rl_weights, dtype=torch.float).unsqueeze(0)
            if micro_batch.rl_weights is not None
            else None,
            ce_weights=torch.tensor(micro_batch.ce_weights, dtype=torch.float).unsqueeze(0)
            if micro_batch.ce_weights is not None
            else None,
            ref_kl_weights=torch.tensor(micro_batch.ref_kl_weights, dtype=torch.float).unsqueeze(0)
            if micro_batch.ref_kl_weights is not None
            else None,
            run_id=micro_batch.run_id,
            run_step=micro_batch.run_step,
        )


def _torch_dtype(name: str) -> torch.dtype:
    """Resolve a numpy/torch dtype name (e.g. ``"float32"``) to torch.dtype."""
    # Strip the ``numpy.`` prefix some dtype reprs carry.
    name = name.replace("numpy.", "")
    if hasattr(torch, name):
        return getattr(torch, name)
    # numpy ↔ torch alias mismatches (rare but possible) — fall back via numpy.
    import numpy as np

    return torch.from_numpy(np.zeros(1, dtype=np.dtype(name))).dtype


class _TraceStep(msgspec.Struct, forbid_unknown_fields=True):
    step_index: int
    batch_id: str
    warmup: bool
    logical_samples: int
    prepared_samples: int
    logical_tokens: int
    prepared_tokens: int
    loss_tokens: int


class _TraceManifest(msgspec.Struct):
    version: int
    artifact_id: str
    candidate_id: str
    runtime: str
    tokenized_artifact_id: str
    converter_revision: str
    temperature: float
    padding_multiple: int
    steps: list[_TraceStep]
    row_counts: dict[str, int]


class _TraceRecord(msgspec.Struct, forbid_unknown_fields=True):
    step_index: int
    sample_id: str
    input_ids: list[int]
    loss_mask: list[bool]
    advantage: float
    rollout_logprobs: list[float]


class _TraceStepReader:
    def __init__(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(f"Prepared Prime steps not found: {path}")

        self.manifest = _load_trace_manifest(path.parent / "manifest.json")
        self._file: IO[str] = gzip.open(path, "rt", encoding="utf-8")
        self._decoder = msgspec.json.Decoder(type=_TraceRecord)
        self._line_number = 0
        self._record_count = 0
        self._next_step = 0
        self._pending: _TraceRecord | None = None

    def read_step(self) -> list[TrainingSample]:
        if self._next_step >= 6:
            raise ValueError("Prepared Prime artifact contains only six steps")

        expected_step = self._next_step
        record = self._pending or self._read_record()
        self._pending = None
        if record is None:
            raise ValueError(f"Prepared Prime artifact is missing step {expected_step}")
        if record.step_index != expected_step:
            raise ValueError(
                f"Prepared Prime step order is invalid: expected {expected_step}, found {record.step_index}"
            )

        samples: list[TrainingSample] = []
        while record is not None and record.step_index == expected_step:
            samples.append(_to_training_sample(record, self.manifest.temperature))
            record = self._read_record()

        summary = self.manifest.steps[expected_step]
        actual = (
            len(samples),
            sum(len(sample.token_ids) for sample in samples),
            sum(sum(sample.mask) for sample in samples),
        )
        expected = (summary.prepared_samples, summary.prepared_tokens, summary.loss_tokens)
        if actual != expected:
            raise ValueError(
                f"Prepared Prime step {expected_step} does not match its manifest: "
                f"expected samples/tokens/loss_tokens {expected}, found {actual}"
            )

        if record is not None:
            if record.step_index != expected_step + 1:
                raise ValueError(
                    f"Prepared Prime step order is invalid: expected {expected_step + 1}, found {record.step_index}"
                )
            self._pending = record

        self._next_step += 1
        if expected_step == 5:
            self.close()
            expected_records = self.manifest.row_counts["records"]
            if self._record_count != expected_records:
                raise ValueError(
                    f"Prepared Prime record count mismatch: expected {expected_records}, found {self._record_count}"
                )
        elif self._pending is None:
            raise ValueError(f"Prepared Prime artifact ended after step {expected_step}")

        return samples

    def close(self) -> None:
        self._file.close()

    def _read_record(self) -> _TraceRecord | None:
        line = self._file.readline()
        if not line:
            return None
        self._line_number += 1
        try:
            record = self._decoder.decode(line)
        except msgspec.DecodeError as error:
            raise ValueError(f"Invalid prepared Prime record on line {self._line_number}: {error}") from error
        _validate_trace_record(record, self._line_number)
        self._record_count += 1
        return record


class _TracePacker(BasePacker):
    def __init__(
        self,
        trace_path: Path,
        dp_world_size: int,
        seq_len: int,
        pad_to_multiple_of: int,
        transport_config: TransportConfig,
        bin_cost: Callable[[Sequence[int]], int],
        start_step: int,
    ):
        super().__init__(
            dp_world_size,
            seq_len,
            pad_to_multiple_of,
            transport_config,
            bin_cost,
            start_step,
        )
        self.reader = _TraceStepReader(trace_path)

    def pack(self) -> None:
        self._heartbeat()
        samples = self.reader.read_step()
        longest_sample = max(len(sample.token_ids) for sample in samples)
        if longest_sample > self.seq_len:
            raise ValueError(f"Prepared Prime sample length {longest_sample} exceeds model.seq_len {self.seq_len}")
        micro_batch_grid = prepare_batch(
            rollouts=samples,
            seq_len=self.seq_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            num_train_workers=self.dp_world_size,
            idxs=[0] * len(samples),
            num_loras=self.multi_run_manager.max_runs,
            bin_cost=self.bin_cost,
        )
        self.sender.send(micro_batch_grid)


def _load_trace_manifest(path: Path) -> _TraceManifest:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared Prime manifest not found: {path}")
    try:
        manifest = msgspec.json.decode(path.read_bytes(), type=_TraceManifest)
    except msgspec.DecodeError as error:
        raise ValueError(f"Invalid prepared Prime manifest: {error}") from error

    if manifest.version not in {2, 3} or manifest.runtime != "prime":
        raise ValueError("Prepared artifact must be a supported Prime artifact")
    if manifest.temperature <= 0:
        raise ValueError("Prepared Prime temperature must be positive")
    if manifest.padding_multiple != 1:
        raise ValueError("Prepared Prime artifact cannot contain adapter padding")
    if manifest.row_counts.keys() != {"records"}:
        raise ValueError("Prepared Prime row counts must contain records")
    if [step.step_index for step in manifest.steps] != list(range(6)):
        raise ValueError("Prepared Prime artifact must contain steps zero through five")
    if [step.warmup for step in manifest.steps] != [True, False, False, False, False, False]:
        raise ValueError("Prepared Prime artifact must contain one warmup followed by five measured steps")
    if any(step.logical_samples != step.prepared_samples for step in manifest.steps):
        raise ValueError("Prepared Prime steps cannot contain adapter padding")
    if any(step.logical_tokens != step.prepared_tokens for step in manifest.steps):
        raise ValueError("Prepared Prime steps cannot contain token padding")
    if manifest.row_counts["records"] != sum(step.prepared_samples for step in manifest.steps):
        raise ValueError("Prepared Prime record count must match step summaries")
    return manifest


def _validate_trace_record(record: _TraceRecord, line_number: int) -> None:
    length = len(record.input_ids)
    if not record.sample_id:
        raise ValueError(f"Prepared Prime record on line {line_number} has no sample ID")
    if not 0 <= record.step_index <= 5:
        raise ValueError(f"Prepared Prime record on line {line_number} has invalid step {record.step_index}")
    if length == 0:
        raise ValueError(f"Prepared Prime record on line {line_number} has no tokens")
    if any(token_id < 0 for token_id in record.input_ids):
        raise ValueError(f"Prepared Prime record on line {line_number} has a negative token ID")
    if len(record.loss_mask) != length or len(record.rollout_logprobs) != length:
        raise ValueError(f"Prepared Prime token arrays are misaligned on line {line_number}")


def _to_training_sample(record: _TraceRecord, temperature: float) -> TrainingSample:
    length = len(record.input_ids)
    return TrainingSample(
        token_ids=record.input_ids,
        mask=record.loss_mask,
        logprobs=record.rollout_logprobs,
        temperatures=[temperature] * length,
        advantages=[record.advantage] * length,
        env_name="benchmark",
    )
