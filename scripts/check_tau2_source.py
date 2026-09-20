"""Check the hypothesis that the task data is built on the open benchmark tau2-bench.

For each domain, the `<policy>` block of a context from valid.parquet is compared line by line with
the policy file in sierra-research/tau2-bench. If they match, the benchmark's domain databases, tool
schemas and task definitions are available under MIT and can be used to build training data.

    python scripts/check_tau2_source.py                    # downloads the reference files
    python scripts/check_tau2_source.py --reference-dir X  # uses files already downloaded into X
"""

import argparse
import difflib
import hashlib
import json
import urllib.request
from pathlib import Path

from guardian.env import setup
from guardian.io import read_rows
from guardian.parsing import parse_transcript, section, system_turn

ROOT = Path(__file__).resolve().parents[1]
RAW = "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains"
POLICY_FILES = {
    "airline": "airline/policy.md",
    "retail": "retail/policy.md",
    "telecom": "telecom/main_policy.md",
}


def normalize(text: str) -> list[str]:
    return [line.rstrip() for line in text.strip().splitlines() if line.strip()]


def reference(domain: str, reference_dir: Path | None) -> str | None:
    relative = POLICY_FILES[domain]
    if reference_dir is not None:
        path = reference_dir / relative
        return path.read_text(encoding="utf-8") if path.exists() else None
    with urllib.request.urlopen(f"{RAW}/{relative}", timeout=30) as handle:
        return handle.read().decode("utf-8")


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "data/raw/valid.parquet")
    ap.add_argument("--reference-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "results/tau2_match.json")
    args = ap.parse_args()

    rows = read_rows(args.input)
    report = {
        "source": "https://github.com/sierra-research/tau2-bench",
        "license": "MIT",
        "checked_on": "2026-09-20",
        "domains": {},
    }
    for domain, relative in POLICY_FILES.items():
        example = next((r for r in rows if r["id"].startswith(domain)), None)
        if example is None:
            continue
        system = system_turn(parse_transcript(example["prompt"]))
        policy = section(system.text, "policy") if system else None
        upstream = reference(domain, args.reference_dir)
        if policy is None or upstream is None:
            report["domains"][domain] = {"checked": False}
            continue
        ours, theirs = normalize(policy), normalize(upstream)
        report["domains"][domain] = {
            "checked": True,
            "example_id": example["id"],
            "file": relative,
            "lines_in_data": len(ours),
            "lines_upstream": len(theirs),
            "identical": ours == theirs,
            # telecom concatenates main_policy.md with the tech support manual, so the upstream file
            # is a subset of what the agent sees rather than the whole of it
            "upstream_lines_all_present": set(theirs) <= set(ours),
            "line_similarity": round(difflib.SequenceMatcher(None, ours, theirs).ratio(), 4),
            "sha256_in_data": hashlib.sha256("\n".join(ours).encode()).hexdigest()[:16],
            "sha256_upstream": hashlib.sha256("\n".join(theirs).encode()).hexdigest()[:16],
        }

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for domain, m in report["domains"].items():
        if not m["checked"]:
            print(f"{domain:>18}: not checked")
            continue
        if m["identical"]:
            verdict = "identical"
        elif m["upstream_lines_all_present"]:
            verdict = "upstream file fully contained in the context"
        else:
            verdict = f"similarity {m['line_similarity']}"
        print(f"{domain:>18}: {m['lines_in_data']} lines vs {m['lines_upstream']} upstream, {verdict}")


if __name__ == "__main__":
    main()
