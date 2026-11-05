"""Re-evaluate saved GSM8K generations without loading a model or changing original metrics."""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re

from report import group_id

ROOT = Path(__file__).resolve().parent
NUMBER = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
NUMERIC = re.compile(r"(?<![\w.])(" + NUMBER + r")(?![\w.])")
BOXED = re.compile(r"\\boxed\s*\{\s*\\?\$?\s*(" + NUMBER + r")\s*\}")
FINAL_CUE = re.compile(r"(?:final\s+answer|the\s+answer\s+is|therefore|thus|hence|in\s+total|so\s+the\s+answer)", re.I)


def numeric_tokens(text):
    return [(m.group(1), m.start()) for m in NUMERIC.finditer(text)]


def extract(text):
    """Prefer explicit final forms; otherwise use last number as a labeled low-confidence fallback."""
    if "####" in text:
        tail = text.rsplit("####", 1)[1]
        numbers = numeric_tokens(tail)
        if numbers:
            return numbers[-1][0], "marker"
    boxes = BOXED.findall(text)
    if boxes:
        return boxes[-1], "boxed"
    cues = list(FINAL_CUE.finditer(text))
    if cues:
        numbers = numeric_tokens(text[cues[-1].end():])
        if numbers:
            return numbers[-1][0], "final_cue"
    numbers = numeric_tokens(text)
    if numbers:
        return numbers[-1][0], "last_number"
    return None, "none"


def value(number):
    if number is None:
        return None
    try:
        return Decimal(number.replace(",", ""))
    except InvalidOperation:
        return None


def evaluate(predictions):
    rows = []
    for row in predictions:
        gold_text, _ = extract(row["reference"])
        # Every GSM8K reference must contain a machine-readable final answer.
        if "####" not in row["reference"] or gold_text is None:
            raise ValueError(f"Invalid gold answer at test_id={row['test_id']}")
        predicted_text, method = extract(row["prediction"])
        gold, predicted = value(gold_text), value(predicted_text)
        rows.append({"test_id": row["test_id"], "gold": str(gold),
                     "prediction": str(predicted) if predicted is not None else None,
                     "method": method, "correct": predicted is not None and predicted == gold,
                     "old_correct": row["correct"]})
    n = len(rows)
    by_method = {method: {"n": sum(r["method"] == method for r in rows),
                          "correct": sum(r["method"] == method and r["correct"] for r in rows)}
                 for method in ["marker", "boxed", "final_cue", "last_number", "none"]}
    return rows, {"n": n, "accuracy": sum(r["correct"] for r in rows)/n,
                  "parsed_rate": sum(r["prediction"] is not None for r in rows)/n,
                  "explicit_rate": sum(r["method"] in ["marker", "boxed", "final_cue"] for r in rows)/n,
                  "by_method": by_method}


def load_json(path):
    return json.loads(path.read_text())


def table(rows):
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(f"{row[c]:.2f}" if isinstance(row[c], float) else str(row[c])
                                      for c in columns) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--group", help="Comparable group ID from report.py")
    args = parser.parse_args()
    groups = {}
    for path in sorted(args.runs.glob("*/results.json")):
        status = path.parent / "status.json"
        if not status.exists() or load_json(status).get("status") != "completed":
            continue
        result = load_json(path)
        groups.setdefault(group_id(result), []).append((path.parent, result))
    if not groups:
        parser.error("No completed runs")
    if args.group is None and len(groups) != 1:
        parser.error(f"Several incomparable groups: {list(groups)}. Select --group.")
    group = args.group or next(iter(groups))
    if group not in groups:
        parser.error(f"Unknown group {group}")
    chosen = {}
    for folder, result in groups[group]:
        chosen[result["config"]["name"]] = (folder, result)
    first_folder = next(iter(chosen.values()))[0]
    entries = [("base", first_folder / "predictions_base.json", next(iter(chosen.values()))[1]["base"], None)]
    entries += [(name, folder / "predictions_tuned.json", result["tuned"], result["resources"])
                for name, (folder, result) in chosen.items()]
    source_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path, _, _ in entries}
    snapshot = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()[:12]
    out = args.runs / f"rescored_{group}_{snapshot}"
    out.mkdir(exist_ok=True)
    summary, full = [], {}
    ids = None
    for name, path, old, resources in entries:
        predictions = load_json(path)
        current_ids = [(r["test_id"], r["question"], r["reference"]) for r in predictions]
        if ids is not None and ids != current_ids:
            raise ValueError(f"Test rows are not aligned: {name}")
        ids = current_ids
        rows, new = evaluate(predictions)
        full[name] = {"source": str(path), "sha256": source_hashes[name], "old": old,
                      "new": new, "rows": rows}
        summary.append({"model": name, "old_accuracy_%": 100*old["accuracy"],
                        "rescored_accuracy_%": 100*new["accuracy"],
                        "PPL": old["perplexity"], "parsed_%": 100*new["parsed_rate"],
                        "explicit_%": 100*new["explicit_rate"],
                        "trainable_params": resources["trainable_parameters"] if resources else "—",
                        "train_min": resources["training_seconds"]/60 if resources else "—",
                        "peak_allocated_GiB": resources["peak_gpu_allocated_gb"] if resources else "—"})
    (out / "rescore_details.json").write_text(json.dumps(full, ensure_ascii=False, indent=2))
    with (out / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    text = ("# Пересчёт качества по сохранённым ответам\n\n"
            "Все строки относятся к одному test GSM8K (1319 задач). Веса и исходные JSON-метрики "
            "не изменены. В старом протоколе требовалась почти точная строка `#### число`; "
            "новый протокол ищет число после `####`, затем `\\boxed{число}`, затем явный вывод "
            "(final answer, therefore, thus и др.), иначе берёт последнее число текста. "
            "Это единое правило для всех моделей. Последнее число — эвристика; она может "
            "ошибаться на незавершённом рассуждении. `explicit_%` показывает долю ответов, "
            "оценённых без этой эвристики. Новая accuracy не эквивалентна официальному GSM8K leaderboard.\n\n"
            + table(summary) + "\n\nМетод извлечения, правильность по каждой задаче и SHA256 исходных файлов "
            "сохранены в `rescore_details.json`. PPL взята из исходного teacher forcing и от парсера не зависит.\n")
    (out / "summary.md").write_text(text)
    print(out / "summary.md")


if __name__ == "__main__":
    main()
