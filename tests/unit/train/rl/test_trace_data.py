import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import prime_rl.trainer.runs as runs
from prime_rl.configs.shared import FileSystemTransportConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.trainer.rl.data import DataLoader, _TraceStepReader
from prime_rl.trainer.utils import build_bin_cost, export_benchmark_json, print_benchmark
from prime_rl.trainer.world import reset_world


def test_trace_steps_load_as_native_training_samples(tmp_path: Path) -> None:
    path = _write_trace_artifact(tmp_path)
    reader = _TraceStepReader(path)

    for step_index in range(6):
        samples = reader.read_step()
        assert len(samples) == 1
        sample = samples[0]
        assert sample.token_ids == [step_index, step_index + 1, step_index + 2]
        assert sample.mask == [False, True, True]
        assert sample.logprobs == [0.0, -0.5, -0.5]
        assert sample.advantages == [0.25, 0.25, 0.25]
        assert sample.temperatures == [0.8, 0.8, 0.8]


def test_data_loader_packs_trace_steps_for_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_trace_artifact(tmp_path)
    output_dir = tmp_path / "output"
    manager = SimpleNamespace(output_dir=output_dir, max_runs=1, synchronize_state=lambda: None)
    monkeypatch.setattr(runs, "_MULTI_RUN_MANAGER", manager)
    reset_world()
    loader = DataLoader(
        output_dir=output_dir,
        start_step=1,
        dp_world_size=1,
        seq_len=8,
        pad_to_multiple_of=1,
        bin_cost=build_bin_cost(None),
        config=FileSystemTransportConfig(),
        trace_path=path,
    )

    loader.wait_for_batch()
    micro_batches = loader.get_batch()

    assert len(micro_batches) == 1
    assert micro_batches[0]["input_ids"].tolist() == [[0, 1, 2]]
    assert micro_batches[0]["loss_mask"].tolist() == [[False, True, True]]


def test_trace_steps_reject_malformed_or_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Prepared Prime steps not found"):
        _TraceStepReader(tmp_path / "missing.jsonl.gz")

    path = _write_trace_artifact(tmp_path, misalign_step=2)
    reader = _TraceStepReader(path)
    reader.read_step()
    with pytest.raises(ValueError, match="token arrays are misaligned on line 3"):
        reader.read_step()


def test_trace_steps_match_manifest_work(tmp_path: Path) -> None:
    path = _write_trace_artifact(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["steps"][1]["loss_tokens"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reader = _TraceStepReader(path)
    reader.read_step()
    with pytest.raises(ValueError, match="step 1 does not match its manifest"):
        reader.read_step()


def test_trace_manifest_requires_one_warmup_step(tmp_path: Path) -> None:
    path = _write_trace_artifact(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["steps"][1]["warmup"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="one warmup followed by five measured steps"):
        _TraceStepReader(path)


def test_trace_manifest_accepts_v3_provenance_fields(tmp_path: Path) -> None:
    path = _write_trace_artifact(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = 3
    manifest["batches"] = [{"id": "admission", "group_ids": ["group-0"]}]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert _TraceStepReader(path).read_step()[0].token_ids == [0, 1, 2]


def test_trace_data_configures_six_benchmark_steps(tmp_path: Path) -> None:
    path = _write_trace_artifact(tmp_path)
    config = TrainerConfig.model_validate({"data": {"trace": {"path": path}}, "bench": {}})

    assert config.max_steps == 6
    assert config.data.trace is not None
    assert config.data.fake is None

    with pytest.raises(ValidationError, match="requires benchmark mode"):
        TrainerConfig.model_validate({"data": {"trace": {"path": path}}})

    with pytest.raises(ValidationError, match="mutually exclusive"):
        TrainerConfig.model_validate(
            {
                "data": {"trace": {"path": path}, "fake": {}},
                "bench": {},
            }
        )


def test_trace_data_preserves_explicit_benchmark_steps(tmp_path: Path) -> None:
    path = _write_trace_artifact(tmp_path)
    config = TrainerConfig.model_validate(
        {
            "data": {"trace": {"path": path}},
            "bench": {},
            "max_steps": 2,
        }
    )

    assert config.max_steps == 2


def test_benchmark_json_contains_five_warmup_excluded_step_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "benchmark.json"
    history = {
        "step": [1, 2, 3, 4, 5, 6],
        "perf/mfu": [10.0] * 6,
        "perf/throughput": [100.0] * 6,
        "time/step": [90.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        "time/forward_backward": [9.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "perf/peak_memory": [20.0] * 6,
        "optim/lr": [1e-6],
    }
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (0, 80 * 1024**3))

    export_benchmark_json(history, output_path)

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["warmup_steps"] == 1
    assert result["step_seconds"] == [10.0, 20.0, 30.0, 40.0, 50.0]
    assert result["actor_step_seconds"] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_benchmark_table_ignores_metrics_with_other_cadences(monkeypatch: pytest.MonkeyPatch) -> None:
    history = {
        "step": [1, 2],
        "perf/mfu": [10.0, 20.0],
        "perf/throughput": [100.0, 200.0],
        "time/step": [90.0, 10.0],
        "perf/peak_memory": [20.0, 30.0],
        "optim/lr": [1e-6],
    }
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (0, 80 * 1024**3))

    print_benchmark(history)


def _write_trace_artifact(root: Path, misalign_step: int | None = None) -> Path:
    manifest = {
        "version": 2,
        "artifact_id": "artifact",
        "candidate_id": "candidate",
        "runtime": "prime",
        "tokenized_artifact_id": "tokenized",
        "converter_revision": "test",
        "temperature": 0.8,
        "padding_multiple": 1,
        "steps": [
            {
                "step_index": step_index,
                "batch_id": f"batch-{max(0, step_index - 1)}",
                "warmup": step_index == 0,
                "logical_samples": 1,
                "prepared_samples": 1,
                "logical_tokens": 3,
                "prepared_tokens": 3,
                "loss_tokens": 2,
            }
            for step_index in range(6)
        ],
        "row_counts": {"records": 6},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    path = root / "steps.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for step_index in range(6):
            logprobs = [0.0, -0.5] if step_index == misalign_step else [0.0, -0.5, -0.5]
            record = {
                "step_index": step_index,
                "sample_id": f"sample-{step_index}",
                "input_ids": [step_index, step_index + 1, step_index + 2],
                "loss_mask": [False, True, True],
                "advantage": 0.25,
                "rollout_logprobs": logprobs,
            }
            handle.write(json.dumps(record) + "\n")
    return path
