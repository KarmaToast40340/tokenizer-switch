"""collect_results.py on small hand-written run folders (no torch needed)."""
import csv
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "collect_results", Path(__file__).resolve().parents[1] / "scripts/collect_results.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)

TRAIN = {"step": 760, "train_bpb": 1.1, "tokens": 49_872_896, "bytes_seen": 1, "tok_per_s": 8700.0}


def make_run(root, name, cfg, log):
    run = root / name
    run.mkdir(parents=True)
    base = {"domain": "code", "token_budget": 50_000_000, "micro_batch": 1, "grad_accum": 64,
            "expected_dropout": 0.1, "eval_split": "validation", "seed": 0, "account": "u1"}
    (run / "config.json").write_text(json.dumps(base | cfg), encoding="utf-8")
    (run / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log), encoding="utf-8")
    return run


def collect_rows(tmp_path, capsys=None):
    out = tmp_path / "runs.csv"
    assert collect.main([str(tmp_path / "runs"), "--out", str(out)]) == 0
    with open(out, encoding="utf-8", newline="") as f:
        return {r["run"]: r for r in csv.DictReader(f)}


def test_nested_runs_headline_and_protocol_columns(tmp_path):
    root = tmp_path / "runs" / "djam_code50_corrected"
    make_run(root, "A", {"condition": "keep"}, [TRAIN, {"final_eval": {"old": 1.0}, "wall_s": 7200}])
    make_run(root, "C", {"condition": "swap", "init": "random"}, [TRAIN, {"final_eval": {"new": 1.3}}])
    make_run(root, "E", {"condition": "progressive", "init": "fvt"},
             [TRAIN, {"final_eval": {"old": 1.5, "new": 1.2}}, {"final_eval_pruned": {"new": 1.21}}])
    rows = collect_rows(tmp_path)
    assert [rows[n]["letter"] for n in "ACE"] == ["A", "C", "E"]
    assert [float(rows[n]["bpb"]) for n in "ACE"] == [1.0, 1.3, 1.21]
    assert all(rows[n]["status"] == "done" for n in "ACE")
    assert (rows["A"]["eval_split"], rows["A"]["micro_batch"], rows["A"]["grad_accum"]) == ("validation", "1", "64")
    assert rows["A"]["dropout"] == "0.1" and rows["A"]["wall_h"] == "2.0"


def test_progressive_run_killed_before_pruned_eval_is_incomplete(tmp_path):
    make_run(tmp_path / "runs", "E", {"condition": "progressive", "init": "fvt"},
             [TRAIN, {"final_eval": {"old": 1.5, "new": 1.2}}])
    row = collect_rows(tmp_path)["E"]
    assert row["status"] == "incomplete" and row["bpb"] == ""


def test_run_without_final_eval_is_incomplete(tmp_path):
    make_run(tmp_path / "runs", "B", {"condition": "swap", "init": "fvt"}, [TRAIN])
    assert collect_rows(tmp_path)["B"]["status"] == "incomplete"


def test_failed_attempts_are_skipped(tmp_path):
    make_run(tmp_path / "runs", "A", {"condition": "keep"}, [TRAIN, {"final_eval": {"old": 1.0}}])
    make_run(tmp_path / "runs", "A.failed1", {"condition": "keep"}, [TRAIN])
    assert list(collect_rows(tmp_path)) == ["A"]


def test_short_and_full_hash_of_one_commit_do_not_warn(tmp_path, capsys):
    full = "6b9db6d" + "0" * 33
    make_run(tmp_path / "runs", "A", {"condition": "keep", "commit": full}, [TRAIN, {"final_eval": {"old": 1.0}}])
    make_run(tmp_path / "runs", "B", {"condition": "swap", "init": "fvt", "commit": "6b9db6d"},
             [TRAIN, {"final_eval": {"new": 1.1}}])
    collect_rows(tmp_path)
    assert "commits" not in capsys.readouterr().err
