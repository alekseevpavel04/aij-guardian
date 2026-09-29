import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

from guardian.extract import decode, encode
from guardian.final import Boosted, Final, rank_against, tree_arrays
from guardian.uncertainty import features, token_parts

ROOT = Path(__file__).resolve().parents[1]


def exported_boosted(x: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[dict, StandardScaler, LGBMClassifier]:
    scaler = StandardScaler().fit(x)
    model = LGBMClassifier(n_estimators=40, num_leaves=5, min_child_samples=5, random_state=seed, verbose=-1)
    model.fit(scaler.transform(x), y)
    trees = [tree_arrays(t["tree_structure"]) for t in model.booster_.dump_model()["tree_info"]]
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "trees": trees}, scaler, model


def test_trees_walked_with_numpy_give_lightgbm_probabilities():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 12))
    y = (x[:, 0] - x[:, 3] + rng.normal(size=200) > 0).astype(int)
    exported, scaler, model = exported_boosted(x, y)
    fresh = rng.normal(size=(50, 12)) * 2
    assert np.allclose(
        Boosted.from_dict(exported).proba(fresh), model.predict_proba(scaler.transform(fresh))[:, 1], atol=1e-12
    )


def test_a_row_is_ranked_against_the_reference_not_against_its_file():
    reference = np.array([0.1, 0.2, 0.3, 0.4])
    assert rank_against(reference, np.array([0.05, 0.25, 0.9])).tolist() == [0.0, 0.5, 1.0]
    assert rank_against(reference, np.array([0.25])).tolist() == [0.5]


def test_the_judge_adds_its_rank_against_the_contexts_at_its_weight():
    class Const:
        def __init__(self, value):
            self.value = value

        def proba(self, x):
            return np.full(len(x), self.value)

    base = Final([Const(0.5)], Const(0.5), np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5, 1.0)
    judged = Final(
        [Const(0.5)], Const(0.5), np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5, 1.0,
        judge_direction=np.array([1.0, -1.0]), reference_j=np.array([-1.0, 0.0, 1.0, 2.0]), weight_j=0.25,
    )
    vectors = np.array([[3.0, 0.0], [0.0, 3.0]])  # judge scores 3 (above every context) and -3 (below every one)
    u = np.zeros((2, 36))
    assert (judged.score(vectors, u) - base.score(vectors, u)).tolist() == [0.25, 0.0]


def test_a_rule_or_the_threshold_labels_a_row():
    model = Final([], None, np.zeros(1), np.zeros(1), 0.5, 1.0)
    assert model.labels(np.array([0.2, 1.2, 0.4]), [0, 0, 1]) == [0, 1, 1]


def test_token_parts_mark_the_tool_name_argument_names_and_values():
    turn = '⟦ASSISTANT · ход 2⟧\nПроверю.\n\t→ TOOL_CALL get_order: {"order_id": "W123"}'
    spans = [(turn.index("Проверю"), turn.index("Проверю") + 7), (turn.index("get_order"), turn.index("get_order") + 3)]
    spans += [(turn.index("order_id"), turn.index("order_id") + 8), (turn.index('"W123"'), turn.index('"W123"') + 6)]
    spans += [(turn.index("{"), turn.index("{") + 1)]
    assert token_parts(turn, spans) == ["text", "name", "key", "value", "layout"]


def test_uncertainty_features_are_36_numbers_even_without_calls():
    record = {"logp": [-0.1, -2.0, -0.01], "entropy": [0.5, 2.0, 0.1], "margin": [0.9, 0.1, 0.99]}
    x = features(record, ["text", "text", "layout"])
    assert x.shape == (36,) and np.isfinite(x).all()


def test_vectors_survive_the_journal_encoding():
    v = np.random.default_rng(0).normal(size=16).astype(np.float32)
    assert np.array_equal(decode(encode(v)), v)


FAKE_PASS = """
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, {src!r})
import numpy as np
from guardian import journal
from guardian.extract import encode
ap = argparse.ArgumentParser()
for a in ("--rows", "--prefix", "--model-dir", "--system-prompt", "--out-dir", "--max-model-len",
          "--gpu-memory-utilization", "--stop-at", "--engine-json"):
    ap.add_argument(a)
args = ap.parse_args()
out = Path(args.out_dir)
rows = [json.loads(line) for line in open(args.rows, encoding="utf-8")]
for row in rows[:{scored}]:
    v = np.full(4, float(len(row["response"])), dtype=np.float32)
    journal.append(out / f"{{args.prefix}}_final_org.jsonl", {{"id": row["id"], "vector": encode(v), "dim": 4}})
    journal.append(out / f"{{args.prefix}}_parts_org.jsonl", {{"id": row["id"], "parts": ["text"]}})
    journal.append(out / f"{{args.prefix}}_uncertainty_org.jsonl",
                   {{"id": row["id"], "logp": [-1.0], "entropy": [1.0], "margin": [0.5]}})
"""


def final_archive(tmp_path: Path, scored: int, share: float | None = None) -> tuple[Path, Path]:
    """The submission's scripts and package with heads of a toy shape: head B reads a vector of 4 equal numbers
    (the response length), so the longer response scores higher; the uncertainty head is flat."""
    root = tmp_path / "archive"
    (root / "scripts").mkdir(parents=True)
    (root / "model").mkdir()
    shutil.copy(ROOT / "scripts/predict.py", root / "scripts/predict.py")  # the package comes from PYTHONPATH
    x = np.repeat(np.arange(40, dtype=float)[:, None], 4, axis=1)
    y = (np.arange(40) >= 20).astype(int)
    heads = [exported_boosted(x, y, seed)[0] for seed in range(3)]
    b = sum(Boosted.from_dict(h).proba(x) for h in heads)
    final = {
        "head_b": heads,
        "head_u": {"mean": [0.0] * 36, "scale": [1.0] * 36, "coef": [0.0] * 36, "intercept": 0.0},
        "reference_b": sorted(b.tolist()),
        "reference_u": [0.5] * 40,
        "weight_u": 0.5,
        # head B gives the 20 short and the 20 long reference rows two tied scores: a long response ranks above the
        # 20 short ones (0.5), a short one above none (0.0)
        "threshold": 0.4,
        "checks": [],
    }
    (root / "model/final.json").write_text(json.dumps(final))
    (root / "model/judge_prompt.md").write_text("judge")
    config = {
        "strategy": "final",
        "final": "final.json",
        "final_minutes": 2,
        "batch_minutes": 0,
        "max_model_len": 1000,
        "gpu_memory_utilization": 0.5,
        "checks": [],
        "prompt": "judge_prompt.md",
    }
    if share is not None:
        config["share"] = share
    (root / "model/guardian.json").write_text(json.dumps(config))
    fake = tmp_path / "fake_pass.py"
    fake.write_text(FAKE_PASS.format(src=str(ROOT / "src"), scored=scored))
    return root, fake


def run_predict(tmp_path: Path, scored: int, responses: list[str], share: float | None = None) -> list[int]:
    import pandas as pd

    root, fake = final_archive(tmp_path, scored, share)
    rows = pd.DataFrame(
        {"id": [f"r{i}" for i in range(len(responses))], "prompt": ["p"] * len(responses), "response": responses}
    )
    rows.to_parquet(tmp_path / "test.parquet")
    env = {**os.environ, "GUARDIAN_FINAL_PASS": str(fake), "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/predict.py"),
            "--input",
            str(tmp_path / "test.parquet"),
            "--output",
            str(tmp_path / "out.parquet"),
        ],
        check=True,
        env=env,
        capture_output=True,
        cwd=tmp_path,
    )
    return pd.read_parquet(tmp_path / "out.parquet").label.tolist()


def test_the_final_strategy_labels_rows_by_threshold_on_their_own(tmp_path):
    # responses of 5 and 35 characters: the first scores below the threshold, the second above
    assert run_predict(tmp_path, 2, ["x" * 5, "x" * 35]) == [0, 1]


def test_rows_the_pass_did_not_score_get_the_rules_label(tmp_path):
    assert run_predict(tmp_path, 1, ["x" * 35, "x" * 35]) == [1, 0]


def test_a_pass_that_scored_nothing_falls_back_to_the_rules(tmp_path):
    assert run_predict(tmp_path, 0, ["x" * 35, "x" * 35]) == [0, 0]


def test_a_configured_share_replaces_the_threshold(tmp_path):
    # both rows score below the threshold; the top half of two tied rows takes both (ties at the cut go in)
    assert run_predict(tmp_path, 2, ["x" * 5, "x" * 5]) == [0, 0]
    assert run_predict(tmp_path / "share", 2, ["x" * 5, "x" * 5], share=0.5) == [1, 1]
