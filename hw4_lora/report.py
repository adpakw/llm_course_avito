"""Build a comparison from completed runs. Does not load models or start training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHARED_TRAINING = ["seed", "epochs", "batch_size", "gradient_accumulation_steps", "max_length",
                   "weight_decay", "warmup_ratio", "bf16", "gradient_checkpointing"]


def read(path):
    return json.loads(path.read_text())


def group_id(result):
    value = {"protocol": result["protocol_id"], "train": result["dataset"]["train_sha256"],
             "training": {key: result["config"][key] for key in SHARED_TRAINING}}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(rows):
    if not rows:
        return "Нет завершённых запусков."
    keys = list(rows[0])
    def fmt(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"]*len(keys)) + " |"] +
                     ["| " + " | ".join(fmt(row[key]) for key in keys) + " |" for row in rows])


def metric_row(name, metrics, config=None, resources=None):
    config, resources = config or {}, resources or {}
    return {"model": name, "accuracy": metrics["accuracy"], "PPL": metrics["perplexity"],
            "MAE_parsed": metrics["mae_parsed"], "parse_rate": metrics["parse_rate"],
            "r": config.get("r"), "alpha": config.get("alpha"), "dropout": config.get("dropout"),
            "target_modules": ", ".join(config.get("target_modules", [])),
            "learning_rate": config.get("learning_rate"),
            "trainable_parameters": resources.get("trainable_parameters"),
            "train_seconds": resources.get("training_seconds"),
            "peak_allocated_GiB": resources.get("peak_gpu_allocated_gb"),
            "peak_reserved_GiB": resources.get("peak_gpu_reserved_gb")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--group", help="ID сопоставимой группы, если менялись общие настройки")
    args = parser.parse_args()
    groups = {}
    for path in sorted(args.runs.glob("*/results.json")):
        status = path.parent / "status.json"
        if not status.exists() or read(status).get("status") != "completed":
            continue
        result = read(path)
        groups.setdefault(group_id(result), []).append((path.parent, result))
    if not groups:
        print("Нет завершённых запусков. Сначала запустите full_sft.py и lora.py.")
        return
    if args.group is None and len(groups) > 1:
        print("Найдены несопоставимые группы; выберите --group:")
        for key, runs in groups.items():
            print(key, [r["run"] for _, r in runs])
        raise SystemExit(2)
    selected = args.group or next(iter(groups))
    if selected not in groups:
        parser.error(f"Неизвестная группа: {selected}")
    # Latest completed run per name; retain every source run in runs/comparison.json.
    chosen = {}
    for folder, result in groups[selected]:
        chosen[result["config"]["name"]] = (folder, result)
    first = next(iter(chosen.values()))[1]
    rows = [metric_row("base", first["base"])]
    for name, (_, result) in chosen.items():
        if result["base"] != first["base"]:
            raise ValueError("Baseline differs within the same evaluation protocol")
        rows.append(metric_row(name, result["tuned"], result["config"], result["resources"]))
    source_paths = {name: str(folder.resolve()) for name, (folder, _) in chosen.items()}
    snapshot = hashlib.sha256(json.dumps(source_paths, sort_keys=True).encode()).hexdigest()[:8]
    # A new training run gets a new report folder, so old human grades never attach to new responses.
    out = args.runs / f"report_{selected}_{snapshot}"
    out.mkdir(exist_ok=True)
    write_csv(out / "comparison.csv", rows)
    text = ("# Результаты Full FT и LoRA\n\n"
            f"Группа: `{selected}`. Последний завершённый запуск каждого имени. "
            "Accuracy измеряется от 0 до 1; память — в GiB. Пустые поля не означают нули.\n\n" + table(rows))
    effects = []
    if "lora_default" in chosen:
        baseline = chosen["lora_default"][1]
        for name, (_, result) in chosen.items():
            if name == "lora_default" or "r" not in result["config"]:
                continue
            differences = [key for key in result["config"]
                           if key != "name" and result["config"][key] != baseline["config"].get(key)]
            if len(differences) != 1 or differences[0] not in ["r", "alpha", "dropout", "target_modules"]:
                raise ValueError(f"{name}: не однофакторный эксперимент относительно lora_default: {differences}")
            ppl, base_ppl = result["tuned"]["perplexity"], baseline["tuned"]["perplexity"]
            effects.append({"experiment": name, "changed": differences[0],
                            "delta_accuracy_pp": 100*(result["tuned"]["accuracy"]-baseline["tuned"]["accuracy"]),
                            "delta_PPL": ppl-base_ppl if ppl is not None and base_ppl is not None else None})
    if effects:
        text += "\n\n## Изменения относительно базовой LoRA\n\n" + table(effects)
        write_csv(out / "lora_effects.csv", effects)
    if "full_sft" in chosen and "lora_default" in chosen:
        full = chosen["full_sft"][1]["resources"]
        lora = chosen["lora_default"][1]["resources"]
        ratios = [{"metric": key, "full_over_lora": full[key]/lora[key]} for key in
                  ["trainable_parameters", "training_seconds", "peak_gpu_allocated_gb"]]
        text += "\n\n## Отношение затрат Full FT / LoRA\n\n" + table(ratios)
        write_csv(out / "resource_ratios.csv", ratios)
        full_dir, lora_dir = chosen["full_sft"][0], chosen["lora_default"][0]
        base_predictions = read(full_dir / "predictions_base.json")
        full_predictions = read(full_dir / "predictions_tuned.json")
        lora_predictions = read(lora_dir / "predictions_tuned.json")
        count = min(chosen["full_sft"][1]["config"]["manual_examples"],
                    chosen["lora_default"][1]["config"]["manual_examples"])
        manual, grades = ["# Общие задачи для ручной оценки\n"], []
        for index in range(count):
            base = base_predictions[index]
            manual.append(f"\n## test_id={index}\n\n{base['question']}\n\nЭталон:\n\n{base['reference']}\n")
            for name, prediction in [("base", base), ("full_sft", full_predictions[index]),
                                     ("lora_default", lora_predictions[index])]:
                if prediction["question"] != base["question"]:
                    raise ValueError("Manual examples are not aligned")
                manual.append(f"\n### {name}\n\n{prediction['prediction']}\n")
                grades.append({"test_id": index, "model": name, "answer_correct_0_1": "",
                               "reasoning_0_1_2": "", "format_0_1": "", "comment": ""})
        (out / "manual_review.md").write_text("\n".join(manual))
        if not (out / "manual_grades.csv").exists():
            write_csv(out / "manual_grades.csv", grades)
    (out / "sources.json").write_text(json.dumps(source_paths, ensure_ascii=False, indent=2))
    text += "\n\nВыводы и ручные оценки заполните после изучения этих результатов; методика — в REPORT.md.\n"
    (out / "summary.md").write_text(text)
    print(f"Отчёт: {out / 'summary.md'}")


if __name__ == "__main__":
    main()
