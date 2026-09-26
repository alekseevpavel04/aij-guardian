"""How close each writer's messages to the user are to the organiser's agent, by surface style.

The method and the regular expressions come from the author's detector project (prompt_bot,
`корпус/таблица_нграмм.py`, functions `предложения()` and `строение()`), cut down to what a message of
100–800 characters can carry. A message is the agent's text to the user in one turn (call lines dropped),
kept when it has at least five words. Writers: the organiser's agent (its turns in the contexts of
airline, retail and telecom), each candidate of agent_fingerprint.py, and each simulation's agent
(rows files of simulate_tau2.py / build_pseudo_rows.py).

Per message: length, sentences, words per sentence, commas, long dashes and colons per 1000 characters,
long words (over 11 letters), share of anchor characters (digits, Latin, «quotes», initials), digits,
and whether it has bold, a list, a dollar sign. Rare markers (introductory words, subordinating
connectives, verbal nouns, participles) are counted over all of a writer's messages per 1000 words.
Against the organiser, as in prompt_bot: d = (writer's median − organiser's median) / organiser's pstdev,
and the share of the writer's messages outside the organiser's 10–90 % range (20 % if they match);
a feature is marked when |d| ≥ 0.5 or that share ≥ 40 %. Pooled rates are given as a ratio to the
organiser's with a bootstrap interval over messages.

    python scripts/style_compare.py
"""

import argparse
import json
import re
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from guardian import journal
from guardian.env import setup

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("airline", "retail", "telecom")

# from prompt_bot/корпус/таблица_нграмм.py
WORD = re.compile(r"[А-Яа-яЁёA-Za-z]+")
ABBREVIATIONS = re.compile(
    r"\b[А-ЯЁA-Z]\.(?:\s?[А-ЯЁA-Z]\.)*|"
    r"\b(?:т|д|е|п|г|гг|в|вв|рис|табл|см|стр|с|ст|проф|доц|акад|им|обл|"
    r"руб|тыс|млн|млрд|шт|кв|куб|мин|макс|ср|напр|др|пр)\.",
    re.I,
)
SHIELD = ""
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
INTRODUCTORY = re.compile(
    r"\b(впрочем|пожалуй|разумеется|конечно|видимо|по-видимому|кстати|"
    r"однако|между тем|в частности|в сущности|по существу|"
    r"стало быть|следовательно|таким образом|итак|например|скажем|"
    r"прежде всего|кроме того|более того|тем не менее|вместе с тем|"
    r"с одной стороны|с другой стороны|как известно|на мой взгляд)\b",
    re.I,
)
CONNECTIVES = re.compile(
    r"\b(который|которая|которое|которые|которых|которым|поскольку|"
    r"тогда как|при котором|при наличии|чем|причём|благодаря|вследствие|"
    r"несмотря на|хотя|если|когда|пока|чтобы|так как|потому что)\b",
    re.I,
)
VERBAL_NOUNS = re.compile(r"\b\w+(?:ание|ение|ания|ения|аний|ений|анию|ению|анием|ением)\b", re.I)
PARTICIPLES = re.compile(
    r"\b\w+(?:ющий|ющая|ющее|ющие|ющих|ющим|вший|вшая|вшее|вшие|вших|нный|нная|нное|нные|нных|емый|имый|ащий|ящий)\b",
    re.I,
)
ANCHOR = re.compile(r"\d|[A-Za-z]|«[^»]{1,40}»|\b[А-ЯЁ]\.\s?[А-ЯЁ]\.")
LIST_LINE = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s", re.MULTILINE)
POOLED = {
    "introductory": INTRODUCTORY,
    "connectives": CONNECTIVES,
    "verbal_nouns": VERBAL_NOUNS,
    "participles": PARTICIPLES,
}


def sentences(text: str) -> list[str]:
    shielded = ABBREVIATIONS.sub(lambda m: m.group(0).replace(".", SHIELD), text)
    return [s.replace(SHIELD, ".").strip() for s in SENTENCE_END.split(shielded) if s.strip()]


def prose(turn: str) -> str:
    lines = turn.split("\n", 1)[1].splitlines() if turn.startswith("⟦") and "\n" in turn else turn.splitlines()
    return "\n".join(line for line in lines if not line.strip().startswith(("→", "←"))).strip()


def features(text: str) -> dict[str, float] | None:
    words = WORD.findall(text)
    if len(words) < 5:
        return None
    lengths = [len(WORD.findall(s)) for s in sentences(text)]
    lengths = [n for n in lengths if n] or [len(words)]
    k = len(text) / 1000
    return {
        "chars": len(text),
        "sentences": len(lengths),
        "words_per_sentence": statistics.mean(lengths),
        "commas_per_1k": text.count(",") / k,
        "long_dashes_per_1k": (text.count("—") + text.count("–")) / k,
        "colons_per_1k": text.count(":") / k,
        "long_words_pct": sum(len(w) > 11 for w in words) / len(words) * 100,
        "anchor_chars_pct": sum(len(m.group(0)) for m in ANCHOR.finditer(text)) / len(text) * 100,
        "digits_pct": sum(c.isdigit() for c in text) / len(text) * 100,
        "has_bold": float("**" in text),
        "has_list": float(bool(LIST_LINE.search(text))),
        "has_dollar_sign": float("$" in text),
    }


def pooled_rates(texts: list[str]) -> dict[str, float]:
    words = sum(len(WORD.findall(t)) for t in texts) / 1000
    return {name: sum(len(rx.findall(t)) for t in texts) / words for name, rx in POOLED.items()}


def writers(args: argparse.Namespace) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    contexts = pd.read_parquet(args.contexts)
    contexts = contexts[contexts.domain.isin(DOMAINS)]
    out["organiser"] = [prose(r) for r in contexts.response]
    for path in sorted((ROOT / "data/interim/fingerprint").glob("*.jsonl")):
        out[f"fingerprint:{path.stem}"] = [prose(r["generated"]) for r in journal.read(path) if "generated" in r]
    for name, path in (
        ("simulation:qwen3_8b", ROOT / "data/synthetic/tau2_ru_rows.parquet"),
        ("simulation:qwen36_thinking", ROOT / "data/synthetic/tau2_ru_q36_rows.parquet"),
        ("simulation:glm4_7_flash", ROOT / "data/synthetic/tau2_ru_glm_rows.parquet"),
    ):
        if path.exists():
            out[name] = [prose(r) for r in pd.read_parquet(path).response]
    return {k: [t for t in v if features(t)] for k, v in out.items()}


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=ROOT / "data/synthetic/organizer_contexts.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results/style_compare.json")
    args = ap.parse_args()

    texts = writers(args)
    frames = {k: pd.DataFrame([features(t) for t in v]) for k, v in texts.items()}
    org = frames["organiser"]
    org_pooled = pooled_rates(texts["organiser"])
    rng = np.random.default_rng(0)
    result: dict = {
        "organiser": {"messages": len(org), "medians": org.median().round(2).to_dict(), "pooled": org_pooled}
    }
    for name, frame in frames.items():
        if name == "organiser":
            continue
        per_feature = {}
        for col in org.columns:
            spread = statistics.pstdev(org[col]) or 1.0
            low, high = np.percentile(org[col], [10, 90])
            outside = float(((frame[col] < low) | (frame[col] > high)).mean())
            d = (frame[col].median() - org[col].median()) / spread
            if col.startswith("has_"):  # a share, not a median, for the yes/no features
                d = (frame[col].mean() - org[col].mean()) / spread
            per_feature[col] = {
                "writer": round(float(frame[col].mean() if col.startswith("has_") else frame[col].median()), 2),
                "d": round(float(d), 2),
                "outside_10_90": round(outside, 2),
                "marked": bool(abs(d) >= 0.5 or outside >= 0.4),
            }
        pooled = {}
        for key, value in pooled_rates(texts[name]).items():
            draws = []
            for _ in range(200):
                pick = [texts[name][i] for i in rng.integers(0, len(texts[name]), len(texts[name]))]
                draws.append(pooled_rates(pick)[key] / org_pooled[key] if org_pooled[key] else np.nan)
            pooled[key] = {
                "ratio_to_organiser": round(value / org_pooled[key], 2) if org_pooled[key] else None,
                "ci95": [round(float(np.nanpercentile(draws, 2.5)), 2), round(float(np.nanpercentile(draws, 97.5)), 2)],
            }
        result[name] = {
            "messages": len(frame),
            "mean_abs_d": round(float(np.mean([abs(v["d"]) for v in per_feature.values()])), 3),
            "marked_features": sum(v["marked"] for v in per_feature.values()),
            "features": per_feature,
            "pooled": pooled,
        }
        summary = result[name]
        print(f"{name:40s} n={len(frame):4d} mean|d|={summary['mean_abs_d']:.3f} marked={summary['marked_features']}")
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
