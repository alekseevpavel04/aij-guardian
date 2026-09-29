"""The README figures, drawn from committed results only (and, for the token example, from our own simulation).

architecture.png  the final system: one pass of the model, two heads, ranks, a threshold, the rules
tokens.png        how surprised the model is by each token of two erroneous simulated turns
journey.png       F1 of the ten submissions in order, against the organiser's baseline and the first place
leaderboard.png   precision and recall of the top ten and of our submissions, with lines of equal F1
gap.png           AUROC at home (our labels) against the AUROC the test implies
tried.png         paired differences on the agreed contexts for what was tried, with 95 % intervals

python scripts/readme_figures.py
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

BLUE, ORANGE, DARK, GRAY, LIGHT = "#2563eb", "#ea580c", "#0f172a", "#94a3b8", "#e2e8f0"
GREEN, RED, SOFT = "#16a34a", "#dc2626", "#f8fafc"
CHOSEN = {5, 6, 10}  # DECISIONS 49

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": DARK,
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def subtitle(ax, text: str) -> None:
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=9, color="#64748b", va="bottom")


def save(fig, out: Path) -> None:
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(out)


def submissions() -> list[dict]:
    with (RESULTS / "submissions.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def box(ax, x, y, w, h, title, body="", edge=BLUE, fill="white", dashed=False, title_size=11):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.6,
            edgecolor=edge,
            facecolor=fill,
            linestyle="--" if dashed else "-",
        )
    )
    ax.text(x + 0.18, y + h - 0.2, title, fontsize=title_size, fontweight="bold", color=DARK, va="top")
    if body:
        ax.text(x + 0.18, y + h - 0.58, body, fontsize=8.6, color="#334155", va="top", linespacing=1.35)


def arrow(ax, a, b, color="#475569", dashed=False, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color=color,
            linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def architecture(out: Path) -> None:
    timing = load("rental_timing.json")
    speed = f"{timing['pass_tokens_per_second']:,}".replace(",", " ")
    fig, ax = plt.subplots(figsize=(15, 6.4))
    ax.set_xlim(0, 15.4)
    ax.set_ylim(0, 6.7)
    ax.axis("off")

    def lines(*parts: str) -> str:
        return "\n".join(parts)

    box(
        ax,
        0.1,
        2.2,
        2.85,
        2.3,
        "Ход агента",
        lines(
            "prompt: инструкции,",
            "политика, схемы",
            "инструментов, диалог,",
            "ответы инструментов",
            "",
            "response: текст и вызовы",
        ),
        edge=DARK,
        fill=SOFT,
    )
    box(
        ax,
        3.55,
        4.85,
        3.45,
        1.5,
        "8 проверок",
        lines("инструмент не из списка, чужой", "аргумент, повтор упавшего вызова,", "ID, которого нет в контексте"),
        edge=GRAY,
    )
    box(
        ax,
        3.55,
        1.25,
        3.25,
        3.1,
        "Qwen3.6-35B-A3B",
        lines(
            "FP8, vLLM, как на платформе",
            "",
            "один проход по промпту судьи",
            "организатора, состояния",
            "всех позиций",
            "",
            f"{speed} токенов/с на A100",
            f"{timing['total_seconds']:.0f} с на {timing['rows']} строк",
            "с запуском движка",
        ),
        edge=BLUE,
        fill="#eff6ff",
        title_size=12,
    )
    box(
        ax,
        7.5,
        3.1,
        3.05,
        1.25,
        "Голова B",
        lines("LightGBM 300×7 на состоянии", "последнего токена (2048)"),
        edge=BLUE,
    )
    box(
        ax,
        7.5,
        1.6,
        3.05,
        1.25,
        "Голова неуверенности",
        lines("логистическая, 36 признаков:", "log p, энтропия, зазор по частям"),
        edge=BLUE,
    )
    box(
        ax, 7.5, 0.25, 3.05, 1.1, "Судья", lines("логит «1» против «0»", "из того же состояния"), edge=GRAY, dashed=True
    )
    box(
        ax,
        11.2,
        1.3,
        2.3,
        2.65,
        "Ранги",
        lines(
            "против оценок",
            "384 контекстов",
            "вне фолда",
            "",
            "B + 0.5·U (+ 0.25·J)",
            "",
            "строка решается сама,",
            "не как доля файла",
        ),
        edge=ORANGE,
        fill="#fff7ed",
    )
    box(ax, 13.95, 2.2, 1.35, 0.9, "порог", edge=ORANGE)
    box(ax, 13.95, 5.0, 1.35, 1.15, "ИЛИ", "→ метка 0 / 1", edge=DARK, fill=SOFT)

    arrow(ax, (2.95, 3.9), (3.55, 5.3))
    arrow(ax, (2.95, 3.0), (3.55, 2.7))
    arrow(ax, (6.8, 3.7), (7.5, 3.7))
    arrow(ax, (6.8, 2.2), (7.5, 2.2))
    arrow(ax, (6.8, 1.45), (7.5, 0.8), dashed=True, color=GRAY)
    arrow(ax, (10.55, 3.7), (11.2, 3.25))
    arrow(ax, (10.55, 2.2), (11.2, 2.55))
    arrow(ax, (10.55, 0.8), (11.2, 1.6), dashed=True, color=GRAY)
    arrow(ax, (13.5, 2.65), (13.95, 2.65), color=ORANGE)
    arrow(ax, (14.62, 3.1), (14.62, 5.0), color=ORANGE)
    arrow(ax, (7.0, 5.6), (13.95, 5.6), color=GRAY)
    ax.text(10.35, 5.77, "сработало правило: сразу 1", fontsize=8.6, color="#64748b", ha="center")
    ax.text(0.1, 6.5, "Финальная система (отправки 7–10)", fontsize=14, fontweight="bold", color=DARK)
    ax.text(
        0.1,
        0.02,
        "Пунктир: только отправка 10. Если проход не успел до дедлайна, строка получает метку проверок.",
        fontsize=8.5,
        color="#64748b",
    )
    save(fig, out)


def token_rows(name: str, turn_id: str, model_dir: Path, prompt_path: Path) -> tuple[str, list, np.ndarray]:
    import pandas as pd
    from transformers import AutoTokenizer

    from guardian import journal
    from guardian.extract import render_tokens

    rows = pd.read_parquet(ROOT / f"data/synthetic/tau2_ru_{name}_rows.parquet").set_index("id")
    stats = {r["id"]: r for r in journal.read(ROOT / f"data/interim/embeddings/fp8_sim_{name}_uncertainty_org.jsonl")}
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    row = rows.loc[turn_id]
    rendered = render_tokens(tokenizer, prompt_path.read_text(encoding="utf-8"), row.prompt, row.response)
    logp = np.asarray(stats[turn_id]["logp"])
    parts = {
        r["id"]: r["parts"] for r in journal.read(ROOT / f"data/interim/embeddings/fp8_sim_{name}_parts_org.jsonl")
    }
    if len(logp) != len(rendered.spans) or len(parts[turn_id]) != len(logp):
        raise SystemExit(f"{turn_id}: {len(logp)} probabilities for {len(rendered.spans)} tokens")
    return row.response, rendered.spans, logp, parts[turn_id]


def draw_tokens(ax, text, spans, logp, parts, mark, top, width=14.0, char=0.083, line=0.36):
    """Token chips from left to right, a new row at a line break, coloured by -log p; layout tokens (the turn header,
    call markers, JSON punctuation) grey; the characters `mark` (start, end) of the response underlined in red,
    with the probability range of their tokens above them."""
    cmap = plt.get_cmap("Reds")
    x, y = 0.0, top
    marked = []
    for (a, b), lp, kind in zip(spans, logp, parts, strict=True):
        for j, piece in enumerate(text[a:b].split("\n")):
            if j:
                x, y = 0.0, y - line
            shown = piece.replace("\t", "  ")
            if not shown:
                continue
            w = len(shown) * char + 0.05
            if x + w > width:
                x, y = 0.0, y - line
            s = min(-lp, 8.0) / 8.0
            layout = kind == "layout" and not (mark[0] <= a < mark[1])
            ax.add_patch(
                FancyBboxPatch(
                    (x, y - 0.13),
                    w,
                    0.27,
                    boxstyle="round,pad=0.005,rounding_size=0.05",
                    facecolor="#f1f5f9" if layout else cmap(0.04 + 0.9 * s),
                    edgecolor="none",
                )
            )
            ax.text(
                x + 0.025,
                y,
                shown,
                fontsize=9,
                family="DejaVu Sans Mono",
                va="center",
                color=GRAY if layout else "white" if s > 0.55 else DARK,
            )
            if mark[0] <= a < mark[1]:
                marked.append((x, x + w, y, lp))
            x += w + 0.02
    x0, x1, ym = marked[0][0], marked[-1][1], marked[0][2]
    ax.plot([x0, x1], [ym - 0.19, ym - 0.19], color=RED, lw=2)
    low, high = min(m[3] for m in marked), max(m[3] for m in marked)

    def num(v: float) -> str:
        return f"{v:.1f}".replace("-0.0", "0.0").replace("-", "−")

    label = f"log p = {num(low)}" if len(marked) == 1 else f"log p от {num(low)} до {num(high)}"
    ax.text((x0 + x1) / 2, ym - 0.36, label, ha="center", va="center", fontsize=9, color=RED, fontweight="bold")
    return y - 0.25


def tokens(out: Path, model_dir: Path, prompt_path: Path) -> None:
    telecom = (
        "glm__telecom___mms_issue_airplane_mode_on_bad_network_preference_break_apn_mms_setting_break_app_sms_"
        "permission_data_mode_off_data_usage_exceeded_PERSONA_Easy___trial0::t5"
    )
    examples = [
        (
            "glm",
            telecom,
            "0.5",
            "5",
            "Выдуманный минимум «0.5 ГБ»: такого нет ни в политике, ни в данных",
            "Текстовое утверждение, проверки его не видят. Цифра «5» маловероятна для модели (p ≈ {p}), "
            "но начала обычных фраз вроде «Цена» и «Вы» удивляют её не меньше:\nодин токен ничего не решает, "
            "поэтому голова неуверенности смотрит на сводки по частям хода, а решение принимается вместе с головой B.",
        ),
        (
            "glm",
            "glm__retail__5__trial0::t12",
            "4579334072",
            None,
            "Неверный товар: 4579334072 вместо выбранного пользователем 7843064651",
            "Номер из того же каталога модель считает почти наверняка верным. Такую ошибку вероятности не "
            "показывают: её видно только сверкой с базой\n(в отправках 9 и 10 ходы с вызовами размечены по "
            "эталонным действиям τ²).",
        ),
    ]
    fig, ax = plt.subplots(figsize=(15, 6.2))
    ax.set_xlim(-0.1, 14.3)
    ax.axis("off")
    y = 0.0
    for name, turn_id, needle, digit, title, note in examples:
        text, spans, logp, parts = token_rows(name, turn_id, model_dir, prompt_path)
        start = text.index(needle)
        mark = (start + needle.index(digit), start + needle.index(digit) + 1) if digit else (start, start + len(needle))
        if digit:
            p = float(np.exp(next(lp for (a, _), lp in zip(spans, logp, strict=True) if a == mark[0])))
            note = note.format(p=f"{p * 100:.0f} %")
        ax.text(0, y, title, fontsize=12, fontweight="bold", color=DARK, va="bottom")
        ax.text(0, y - 0.62, note, fontsize=9, color="#475569", va="bottom", linespacing=1.4)
        y = draw_tokens(ax, text, spans, logp, parts, mark, y - 1.1) - 0.75
    ax.set_ylim(y + 0.3, 0.45)
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(0, 8))
    bar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.03, pad=0.02, aspect=60)
    bar.set_label(
        "насколько модель удивлена токеном, −log p (Qwen3.6 FP8, тот же проход, что в решении); служебные токены серые",
        fontsize=9,
    )
    fig.text(
        0.125, 0.975, "Пример: что видят вероятности токенов, а что нет", fontsize=14, fontweight="bold", color=DARK
    )
    fig.text(
        0.125,
        0.945,
        "Два хода с ошибкой из наших симуляций τ² (агент GLM-4.7-Flash, разметка Claude Opus)",
        fontsize=9,
        color="#64748b",
    )
    save(fig, out)


def journey(out: Path) -> None:
    rows = submissions()
    f1 = [float(r["f1"]) for r in rows]
    board = load("budget_model.json")["leaderboard_2026_09_29"]
    forensics = load("leaderboard_forensics.json")
    baseline = next(t for t in forensics["teams"] if t["team"] == "Baseline" and t["f1_leaderboard"] > 0.45)
    always = 2 * forensics["positives_minimal"] / (forensics["test_size_lower_bound"] + forensics["positives_minimal"])
    labels = [
        "правила",
        "судья",
        "судья с\nрассуждением",
        "голова A",
        "голова B",
        "B без пар",
        "финал\nпорог 1.26",
        "финал\nпорог 1.12",
        "финал\nметки τ²",
        "финал\n+ судья",
    ]

    fig, ax = plt.subplots(figsize=(13, 5.6))
    for a, b, day in ((0.5, 2.5, "21.09"), (2.5, 6.5, "23.09"), (6.5, 10.5, "29.09")):
        ax.axvspan(a, b, color=SOFT if day != "23.09" else "#f1f5f9", zorder=0)
        ax.text((a + b) / 2, 0.665, day, ha="center", fontsize=9, color="#64748b")
    for value, text, color, style, shift in (
        (board[0]["f1"], f"1-е место  {board[0]['f1']:.3f}", RED, "--", 0.006),
        (baseline["f1_leaderboard"], f"бейзлайн организатора  {baseline['f1_leaderboard']:.3f}", GRAY, "--", -0.022),
        (always, f"«всегда 1»  {always:.3f}", GRAY, ":", 0.006),
    ):
        ax.axhline(value, color=color, lw=1.1, ls=style, zorder=1)
        ax.text(10.45, value + shift, text, color=color, fontsize=8.5, ha="right")
    x = np.arange(1, len(f1) + 1)
    ax.plot(x, f1, color=BLUE, lw=2, zorder=2)
    for i, v in zip(x, f1, strict=True):
        chosen = i in CHOSEN
        ax.scatter(
            i, v, s=120 if chosen else 60, color=ORANGE if chosen else BLUE, zorder=3, edgecolor="white", linewidth=1.5
        )
        first = i == 1  # the line leaves the first point steeply upwards: its label goes to the right
        ax.text(
            i + (0.12 if first else 0),
            v + (0 if first else 0.022),
            f"{v:.3f}",
            ha="left" if first else "center",
            va="center" if first else "baseline",
            fontsize=9,
            fontweight="bold" if chosen else "normal",
            color=DARK,
        )
    ax.set_xticks(x, [f"{i}\n{s}" for i, s in zip(x, labels, strict=True)], fontsize=8)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.1, 0.69)
    ax.set_ylabel("F1 на публичном тесте")
    ax.set_title("Десять отправок: от правил к головам на состояниях модели", pad=22)
    subtitle(ax, "оранжевые выбраны в финал; F1 из results/submissions.csv")
    ax.grid(axis="y", color=LIGHT, lw=0.8)
    save(fig, out)


def leaderboard(out: Path) -> None:
    budget = load("budget_model.json")
    board = budget["leaderboard_2026_09_29"]
    implied = budget["implied_test_auroc"]
    forensics = load("leaderboard_forensics.json")
    baseline = next(t for t in forensics["teams"] if t["team"] == "Baseline" and t["f1_leaderboard"] > 0.45)
    positives = forensics["positives_minimal"]
    with (RESULTS / "leaderboard_2026-09-29.csv").open(encoding="utf-8") as f:
        teams = list(csv.DictReader(f))
    ours = {  # submissions whose true positives and flags are known
        "1": (13 / positives, 13 / 20),
        "2": (
            implied["cascade_think0"]["tp"] / positives,
            implied["cascade_think0"]["tp"] / implied["cascade_think0"]["k"],
        ),
        "5": (implied["head_b"]["tp"] / positives, implied["head_b"]["tp"] / implied["head_b"]["k"]),
        "7": (
            implied["final_threshold"]["tp"] / positives,
            implied["final_threshold"]["tp"] / implied["final_threshold"]["k"],
        ),
        "8": (
            implied["final_threshold_1.117"]["tp"] / positives,
            implied["final_threshold_1.117"]["tp"] / implied["final_threshold_1.117"]["k"],
        ),
    }

    fig, ax = plt.subplots(figsize=(10, 6.4))
    r = np.linspace(0.02, 1, 400)
    top = 0.7
    for level in (0.3, 0.4, 0.5, 0.6):
        p = level * r / (2 * r - level)
        ok = (p > 0) & (p <= 1)
        ax.plot(r[ok], p[ok], color=LIGHT, lw=1.2, zorder=0)
        ax.text(level * top / (2 * top - level) + 0.008, top, f"F1 {level}", fontsize=8, color=GRAY, va="center")
    for t in teams:
        if t["place"] == "8":  # our submission 5, drawn below
            continue
        ax.scatter(float(t["recall"]), float(t["precision"]), s=90, color="#cbd5e1", edgecolor="white", zorder=3)
        ax.text(float(t["recall"]) + 0.008, float(t["precision"]) + 0.006, t["place"], fontsize=9, color="#475569")
    for n, (rec, prec) in ours.items():
        best = n == "5"
        ax.scatter(rec, prec, s=90 if best else 55, marker="s", color=ORANGE if best else BLUE, zorder=4)
        ax.text(
            rec - 0.012,
            prec - 0.022,
            "№5, 8-е место" if best else f"№{n}",
            fontsize=8.5,
            color=ORANGE if best else BLUE,
            ha="right",
            fontweight="bold" if best else "normal",
        )
    br, bp = (
        baseline["true_positives"] / positives,
        baseline["true_positives"] / (baseline["true_positives"] + baseline["false_positives"]),
    )
    ax.scatter(br, bp, s=120, marker="*", color=DARK, zorder=4)
    ax.text(br + 0.01, bp - 0.02, "бейзлайн организатора", fontsize=8.5, color=DARK)
    ax.scatter([], [], s=90, color="#cbd5e1", label="первая десятка, число это место")
    ax.scatter([], [], s=55, marker="s", color=BLUE, label="наши отправки, где известны точность и полнота")
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0.3, 0.72)
    ax.grid(color="#f1f5f9", lw=0.8)
    ax.set_xlabel("полнота (recall)")
    ax.set_ylabel("точность (precision)")
    ax.set_title("Таблица лидеров 29.09: лидеры точнее при той же полноте", pad=22)
    top4 = [b["auroc"] for b in board[:4]]
    subtitle(
        ax,
        f"оценка AUROC на тесте: первая четвёрка {min(top4):.2f}–{max(top4):.2f}, у нас {board[7]['auroc']:.2f}; "
        "results/leaderboard_2026-09-29.csv, results/budget_model.json",
    )
    save(fig, out)


def gap(out: Path) -> None:
    home = load("ruler_vs_test.json")["systems"]
    final_home = load("paired_final_fp8_ruler__vs__reference.json")["auroc_a"]
    implied = load("budget_model.json")["implied_test_auroc"]
    board = load("budget_model.json")["leaderboard_2026_09_29"]
    rows = [
        (
            "судья организатора (№2)",
            home["judge_no_thinking"]["auroc_v1s_all"],
            implied["cascade_think0"]["auroc"],
            None,
        ),
        ("голова B (№5)", home["head_b"]["auroc_v1s_all"], implied["head_b"]["auroc"], None),
        (
            "голова B без пар (№6)",
            home["head_b_no_pairs"]["auroc_v1s_all"],
            implied["head_b_without_pairs"]["auroc"],
            None,
        ),
        (
            "финальная система (№7, №8)",
            final_home,
            implied["final_threshold"]["auroc"],
            implied["final_threshold_1.117"]["auroc"],
        ),
    ]
    top4 = [r["auroc"] for r in board[:4]]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.axvspan(min(top4), max(top4), color="#fee2e2", zorder=0)
    ax.text(
        (min(top4) + max(top4)) / 2, -0.45, "первая четвёрка\nтаблицы на тесте", ha="center", fontsize=8.5, color=RED
    )
    for i, (name, h, t, t2) in enumerate(rows):
        y = len(rows) - 1 - i
        lo = min(t, t2) if t2 else t
        ax.plot([lo, h], [y, y], color=LIGHT, lw=6, solid_capstyle="round", zorder=1)
        ax.scatter(h, y, s=110, color=BLUE, zorder=3)
        ax.text(h + 0.006, y, f"{h:.3f}", va="center", fontsize=9, color=BLUE)
        if t2:
            ax.plot([t2, t], [y, y], color=ORANGE, lw=3, zorder=2)
        ax.scatter([t] + ([t2] if t2 else []), [y] * (2 if t2 else 1), s=110, color=ORANGE, zorder=3)
        label = f"{min(t, t2):.3f}–{max(t, t2):.3f}" if t2 else f"{t:.3f}"
        ax.text(lo - 0.006, y, label, va="center", ha="right", fontsize=9, color=ORANGE)
        ax.text(0.6, y, name, va="center", fontsize=10, color=DARK)
    ax.scatter([], [], s=80, color=BLUE, label="дома: наши 384 контекста, метки v1s, вне фолда")
    ax.scatter([], [], s=80, color=ORANGE, label="на тесте: оценка по F1 и числу пометок")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, frameon=False, fontsize=8.5)
    ax.set_xlim(0.6, 0.97)
    ax.set_ylim(-0.75, len(rows) - 0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("AUROC")
    ax.set_title("Главный вывод: дома системы неразличимы, на тесте они на 0.15–0.20 ниже", pad=22)
    subtitle(
        ax, "results/ruler_vs_test.json, results/paired_final_fp8_ruler__vs__reference.json, results/budget_model.json"
    )
    save(fig, out)


def tried(out: Path) -> None:
    def paired(name: str) -> tuple[float, float, float]:
        r = load(name)
        return r["a_minus_b"], r["ci95"][0], r["ci95"][1]

    def nested(name: str, *keys: str) -> tuple[float, float, float]:
        r = load(name)
        for k in keys:
            r = r[k]
        return r["a_minus_b"], r["ci95"][0], r["ci95"][1]

    judge = load("judge_signal.json")["paired_against_the_system"]
    rows = [
        ("Данные", None, ""),
        ("симуляции τ² в обучении, против без них", paired("paired_final_fp8_ruler__vs__reference__agreed.json"), ""),
        ("все 9 наборов симуляций вместо 4", nested("data_grid.json", "all", "compared", "base", "agreed"), ""),
        ("Модель и признаки", None, ""),
        ("признаки FP8 (платформа) вместо Q4 (дом)", paired("paired_final_fp8_ruler__vs__grid_base__agreed.json"), ""),
        ("LightGBM 1500×31 вместо 300×7", nested("fp8_settings_check.json", "lgbm_huge", "agreed"), ""),
        ("Gemma 4 26B-A4B вместо Qwen3.6*", nested("model_swap.json", "paired", "system_agreed"), "не отправлялась"),
        ("Сборка", None, ""),
        (
            "стекер на логитах вместо рангов",
            paired("paired_stack_base__vs__stack_base__agreed__both_w1.0_stacked__both_w1.0.json"),
            "",
        ),
        (
            "ранги против контекстов вместо рангов в файле",
            paired("paired_stack_base__vs__stack_base__agreed__both_w1.0_quantile__both_w1.0.json"),
            "отправки 7–10",
        ),
        (
            "судья поверх системы, вес 0.25",
            (judge["system_judge0.25_agreed"]["a_minus_b"], *judge["system_judge0.25_agreed"]["ci95"]),
            "тест: 0.500",
        ),
        (
            "судья поверх системы, вес 1",
            (judge["system_judge1.0_agreed"]["a_minus_b"], *judge["system_judge1.0_agreed"]["ci95"]),
            "",
        ),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    y = len(rows)
    ticks, names = [], []
    for name, value, note in rows:
        y -= 1
        side = ax.get_yaxis_transform()  # x in axes fractions, y in data
        if value is None:
            ax.text(
                -0.02, y, name, transform=side, fontsize=10.5, fontweight="bold", color=DARK, va="center", ha="right"
            )
            continue
        d, lo, hi = value
        color = GREEN if lo > 0 else RED if hi < 0 else GRAY
        ax.plot([lo, hi], [y, y], color=color, lw=2.6, solid_capstyle="round")
        ax.scatter(d, y, s=60, color=color, zorder=3)
        interval = f"{d:+.4f}  [{lo:+.4f}; {hi:+.4f}]".replace("-", "−")
        ax.text(
            1.03, y, interval, transform=side, fontsize=8.8, color="#334155", va="center", family="DejaVu Sans Mono"
        )
        if note:
            ax.text(1.42, y, note, transform=side, fontsize=8.8, color=ORANGE, va="center")
        ticks.append(y)
        names.append(name)
    ax.axvline(0, color=DARK, lw=1)
    ax.set_yticks(ticks, names, fontsize=9.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#f1f5f9", lw=0.8)
    ax.set_xlim(-0.068, 0.022)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("разница AUROC на согласных контекстах (343, 53 ошибки), 95 % интервал бутстрэпа по диалогам")
    ax.set_title("Что пробовали: почти всё внутри шума линейки", pad=22)
    subtitle(ax, "зелёный: интервал выше нуля, красный: ниже; * система только на контекстах, без симуляций и τ²")
    save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs/img")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "data/raw/baseline/baseline/model")
    ap.add_argument(
        "--prompt", type=Path, default=ROOT / "data/raw/baseline/baseline/src/hallucination_detector/prompt.md"
    )
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "architecture": lambda p: architecture(p),
        "tokens": lambda p: tokens(p, args.model_dir, args.prompt),
        "journey": journey,
        "leaderboard": leaderboard,
        "gap": gap,
        "tried": tried,
    }
    for name, draw in figures.items():
        if args.only is None or name in args.only:
            draw(args.out_dir / f"{name}.png")


if __name__ == "__main__":
    main()
