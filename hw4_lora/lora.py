"""LoRA on GSM8K. Edit CONFIG, then run this file; no work on import."""
import argparse
import hashlib
import importlib.metadata
import inspect
import os
import sys
import subprocess
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import re
import time

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainerCallback
from transformers import TrainingArguments
from transformers import set_seed
from transformers.trainer_callback import PrinterCallback
from transformers.trainer_callback import ProgressCallback
from tqdm.auto import tqdm


CONFIG = {
    "name": "lora_default",
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "model_revision": "main",
    "dataset_name": "openai/gsm8k",
    "dataset_revision": "main",
    "seed": 42,
    "epochs": 3,
    "learning_rate": 2e-4,
    "batch_size": 2,
    "eval_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "max_length": 2048,
    "max_new_tokens": 512,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "bf16": True,
    "logging_steps": 1,
    "gradient_checkpointing": True,
    "train_limit": None,
    "test_limit": None,
    "manual_examples": 10,
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
}


LORA_EXPERIMENTS = [
    {"name": "lora_default"},
    {"name": "lora_r8", "r": 8},
    {"name": "lora_r32", "r": 32},
    {"name": "lora_alpha16", "alpha": 16},
    {"name": "lora_alpha64", "alpha": 64},
    {"name": "lora_dropout0", "dropout": 0.0},
    {"name": "lora_dropout01", "dropout": 0.1},
    {"name": "lora_attention", "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
    {"name": "lora_all_projections", "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]},
]


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def resolve_revisions(config):
    # Both scripts share the same pinned source revisions; never track moving main silently.
    path = DATA_DIR / "revisions.json"
    requested = {key: config[key] for key in
                 ["model_name", "model_revision", "dataset_name", "dataset_revision"]}
    if path.exists():
        info = json.loads(path.read_text())
        if info["requested"] != requested:
            raise ValueError("Source configuration changed. Use a separate project/data directory for a new study.")
    else:
        api = HfApi()
        info = {"requested": requested,
                "model_sha": api.model_info(config["model_name"], revision=config["model_revision"]).sha,
                "dataset_sha": api.dataset_info(config["dataset_name"], revision=config["dataset_revision"]).sha}
        save_json(path, info)
    config["model_revision"] = info["model_sha"]
    config["dataset_revision"] = info["dataset_sha"]


def evaluation_protocol(config, tokenizer, test_data):
    keys = ["model_name", "model_revision", "dataset_name", "dataset_revision", "seed",
            "eval_batch_size", "max_new_tokens", "bf16"]
    return {
        "config": {key: config[key] for key in keys},
        "test_sha256": fingerprint(test_data.to_list()),
        "chat_template": tokenizer.chat_template,
        "parameter_dtype": "float32",
        "evaluation_code": fingerprint([inspect.getsource(fn) for fn in
            [evaluate, make_batch, make_messages, prepare_example, get_number]]),
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers"]},
        "gpu": torch.cuda.get_device_name(0),
    }


def evaluate_base(model, tokenizer, test_data, tokenized_test, config, run_dir):
    protocol = evaluation_protocol(config, tokenizer, test_data)
    cache = DATA_DIR / "baseline" / (fingerprint(protocol) + ".json")
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached["protocol"] != protocol or len(cached["predictions"]) != len(test_data):
            raise ValueError("Baseline cache is inconsistent")
        metrics, predictions = cached["metrics"], cached["predictions"]
        save_json(run_dir / "metrics_base.json", metrics)
        save_json(run_dir / "predictions_base.json", predictions)
        logging.info("Использована сохранённая baseline с теми же данными и протоколом.")
    else:
        metrics, predictions = evaluate(model, tokenizer, test_data, tokenized_test, config, run_dir, "base")
        save_json(cache, {"protocol": protocol, "metrics": metrics, "predictions": predictions})
    save_json(run_dir / "evaluation_protocol.json", protocol)
    return metrics, predictions


def get_number(text):
    # Не берем последнее произвольное число: оно может быть частью решения.
    # Принимаем число после #### или ответ, целиком состоящий из числа.
    if "####" in text:
        text = text.rsplit("####", 1)[1].strip()
    else:
        text = text.strip()

    pattern = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    if re.fullmatch(pattern, text) is None:
        return None

    value = float(text.replace(",", ""))
    if not math.isfinite(value):
        return None
    return value


def make_messages(question):
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that solves mathematics problems.",
        },
        {
            "role": "user",
            "content": question + "\nSolve step by step. End with #### followed by the final number.",
        },
    ]


def prepare_example(example, tokenizer, config):
    messages = make_messages(example["question"])
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    messages.append({"role": "assistant", "content": example["answer"]})
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    if input_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Не совпал префикс chat template.")
    if len(input_ids) > config["max_length"]:
        raise ValueError("Пример слишком длинный. Увеличьте max_length в CONFIG.")

    # Loss считается только по ответу ассистента, а не по условию задачи.
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def make_batch(examples, pad_token_id):
    max_length = max(len(example["input_ids"]) for example in examples)
    input_ids = []
    attention_mask = []
    labels = []
    for example in examples:
        padding = max_length - len(example["input_ids"])
        input_ids.append(example["input_ids"] + [pad_token_id] * padding)
        attention_mask.append(example["attention_mask"] + [0] * padding)
        labels.append(example["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }


def evaluate(model, tokenizer, test_data, tokenized_test, config, run_dir, stage):
    batch_size = config["eval_batch_size"]
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("eval_batch_size должен быть положительным целым числом.")
    total_batches = math.ceil(len(test_data) / batch_size)
    logging.info(
        "Оценка %s: %s задач, batch_size=%s, всего %s батчей.",
        stage, len(test_data), batch_size, total_batches,
    )
    model.eval()
    model.config.use_cache = True
    predictions = []
    correct = 0
    errors = []
    total_loss = 0.0
    total_tokens = 0

    # Полоска считает задачи, но обновляется после каждого готового батча.
    with tqdm(
        total=len(test_data),
        desc=f"Test: {stage}",
        unit="задача",
        dynamic_ncols=True,
    ) as progress, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=config["bf16"]):
        for start in range(0, len(test_data), batch_size):
            end = min(start + batch_size, len(test_data))
            examples = [test_data[index] for index in range(start, end)]
            encoded = [tokenized_test[index] for index in range(start, end)]

            # Perplexity тоже считается батчами. Padding в labels равен -100.
            batch = make_batch(encoded, tokenizer.pad_token_id)
            for key in batch:
                batch[key] = batch[key].to("cuda")
            output = model(**batch, use_cache=False)
            token_count = (batch["labels"][:, 1:] != -100).sum().item()
            total_loss += output.loss.item() * token_count
            total_tokens += token_count
            del output, batch

            prompts = []
            for example in examples:
                prompt = tokenizer.apply_chat_template(
                    make_messages(example["question"]),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt)

            # Для батчевой генерации decoder-only модели padding нужен слева.
            # После токенизации возвращаем исходную настройку для обучения.
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
            try:
                inputs = tokenizer(
                    prompts,
                    padding=True,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
            finally:
                tokenizer.padding_side = original_padding_side
            inputs = inputs.to("cuda")
            prompt_length = inputs["input_ids"].shape[1]
            if prompt_length + config["max_new_tokens"] > model.config.max_position_embeddings:
                raise ValueError("Условие и генерация не помещаются в контекст модели.")
            generated = model.generate(
                **inputs,
                max_new_tokens=config["max_new_tokens"],
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=model.generation_config.eos_token_id,
                temperature=None,
                top_p=None,
                top_k=None,
            )
            # Убираем всю входную часть, включая левый padding.
            new_tokens = generated[:, prompt_length:]
            answers = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for offset, (example, answer) in enumerate(zip(examples, answers)):
                predicted_number = get_number(answer)
                gold_number = get_number(example["answer"])
                if gold_number is None:
                    raise ValueError("Не получилось извлечь число из эталонного ответа.")
                is_correct = predicted_number is not None and predicted_number == gold_number
                correct += int(is_correct)
                error = None
                if predicted_number is not None:
                    difference = abs(predicted_number - gold_number)
                    if math.isfinite(difference):
                        error = difference
                        errors.append(error)
                    else:
                        predicted_number = None

                predictions.append({
                    "test_id": start + offset,
                    "question": example["question"],
                    "reference": example["answer"],
                    "prediction": answer,
                    "gold_number": gold_number,
                    "predicted_number": predicted_number,
                    "correct": is_correct,
                    "absolute_error": error,
                })
            progress.set_postfix(
                batch=f"{start // batch_size + 1}/{total_batches}",
                accuracy=f"{correct / end:.3f}",
                refresh=False,
            )
            progress.update(len(examples))
            del inputs, generated, new_tokens

    mae = None
    if errors:
        # Делим до суммирования, чтобы уменьшить риск переполнения суммы.
        mae = sum(error / len(errors) for error in errors)
    nll = total_loss / total_tokens
    perplexity = math.exp(nll) if nll < 709 else None
    metrics = {
        "examples": len(test_data),
        "accuracy": correct / len(test_data),
        "parse_rate": len(errors) / len(test_data),
        "invalid_answers": len(test_data) - len(errors),
        "mae_parsed": mae,
        "mae_all": mae if len(errors) == len(test_data) else None,
        "answer_nll": nll,
        "answer_tokens": total_tokens,
        "perplexity": perplexity,
    }
    save_json(run_dir / f"metrics_{stage}.json", metrics)
    save_json(run_dir / f"predictions_{stage}.json", predictions)
    logging.info("%s: %s", stage, metrics)
    return metrics, predictions


class TrainingLogger(TrainerCallback):
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.progress = None
        self.last_loss = None

    def on_train_begin(self, args, state, control, **kwargs):
        logging.info(
            "Начинается обучение: %s эпох, %s шагов обновления весов. "
            "На один шаг приходится до %s мини-батчей.",
            args.num_train_epochs,
            state.max_steps,
            args.gradient_accumulation_steps,
        )
        self.progress = tqdm(
            total=state.max_steps,
            desc="Обучение",
            unit="шаг",
            dynamic_ncols=True,
            mininterval=0.2,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self.progress is None:
            return
        self.progress.set_postfix(
            epoch=f"{state.epoch:.2f}/{args.num_train_epochs:g}",
            remaining=state.max_steps - state.global_step,
            loss=self.last_loss,
            refresh=False,
        )
        self.progress.update(state.global_step - self.progress.n)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.progress is not None:
            tqdm.write(
                f"Завершена эпоха {state.epoch:.2f}/{args.num_train_epochs:g}. "
                f"Шаги: {state.global_step}/{state.max_steps}."
            )

    def on_train_end(self, args, state, control, **kwargs):
        if self.progress is not None:
            self.progress.close()
        logging.info("Обучение завершено: %s/%s шагов.", state.global_step, state.max_steps)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        row = {
            "step": state.global_step,
            "epoch": state.epoch,
            "metrics": logs,
        }
        # Метрики пишем в файл, чтобы не создавать новую строку в терминале
        # на каждом шаге. Последний loss показывается прямо рядом с полоской.
        if "loss" in logs:
            self.last_loss = round(logs["loss"], 4)
            if self.progress is not None:
                self.progress.set_postfix(
                    epoch=f"{state.epoch:.2f}/{args.num_train_epochs:g}",
                    remaining=state.max_steps - state.global_step,
                    loss=self.last_loss,
                    refresh=True,
                )
        with open(self.run_dir / "training.jsonl", "a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def run_experiment(config):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + config["name"]
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(run_dir / "run.log"), logging.StreamHandler()],
        force=True,
    )
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "status.json", {"status": "started"})
    logging.info("Новый запуск: %s", config["name"])
    logging.info("Папка результатов: %s", run_dir)
    try:
        train_and_evaluate(config, run_dir)
        save_json(run_dir / "status.json", {"status": "completed"})
    except Exception as error:
        logging.exception("Запуск завершился с ошибкой")
        save_json(run_dir / "status.json", {"status": "failed", "error": str(error)})
        raise


def train_and_evaluate(config, run_dir):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 1:
        raise RuntimeError("Выберите одну GPU: CUDA_VISIBLE_DEVICES=0 python ...; не используйте torchrun.")
    if not torch.cuda.is_available():
        raise RuntimeError("Для обучения нужна CUDA GPU.")
    if config["bf16"] and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU не поддерживает bf16. Установите bf16=False для fp32.")
    if not 5 <= config["manual_examples"] <= 10:
        raise ValueError("manual_examples должен быть от 5 до 10.")
    resolve_revisions(config)
    save_json(run_dir / "config.json", config)
    set_seed(config["seed"])
    logging.info("GPU: %s", torch.cuda.get_device_name(0))
    logging.info("[1/7] Загружается датасет %s...", config["dataset_name"])
    stage_start = time.perf_counter()
    dataset = load_dataset(
        config["dataset_name"],
        "main",
        revision=config["dataset_revision"],
        cache_dir=str(DATA_DIR),
    )
    train_data = dataset["train"]
    test_data = dataset["test"]
    logging.info("Датасет загружен за %.1f сек.", time.perf_counter() - stage_start)
    for key in ("train_limit", "test_limit"):
        if config[key] is not None and config[key] < 1:
            raise ValueError(f"{key} должен быть положительным числом или None.")
    if config["train_limit"] is not None:
        train_data = train_data.select(range(min(config["train_limit"], len(train_data))))
    if config["test_limit"] is not None:
        test_data = test_data.select(range(min(config["test_limit"], len(test_data))))
    if len(test_data) < config["manual_examples"]:
        raise ValueError("test_limit должен быть не меньше manual_examples.")
    logging.info("Будут использованы train=%s, test=%s.", len(train_data), len(test_data))
    save_json(run_dir / "dataset.json", {
        "train_size": len(train_data),
        "test_size": len(test_data),
        "train_fingerprint": train_data._fingerprint,
        "test_fingerprint": test_data._fingerprint,
        "train_sha256": fingerprint(train_data.to_list()),
        "test_sha256": fingerprint(test_data.to_list()),
        "examples": train_data.select(range(min(3, len(train_data)))).to_list(),
    })
    logging.info("[2/7] Загружается токенизатор %s...", config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        revision=config["model_revision"],
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    logging.info("Токенизатор загружен.")

    def tokenize_example(example):
        return prepare_example(example, tokenizer, config)

    logging.info("[3/7] Подготовка данных: chat template и токенизация train/test...")
    stage_start = time.perf_counter()
    tokenized_train = train_data.map(
        tokenize_example,
        remove_columns=train_data.column_names,
        desc="Подготовка train",
    )
    tokenized_test = test_data.map(
        tokenize_example,
        remove_columns=test_data.column_names,
        desc="Подготовка test",
    )
    logging.info("Данные подготовлены за %.1f сек.", time.perf_counter() - stage_start)

    def collate(examples):
        return make_batch(examples, tokenizer.pad_token_id)

    logging.info("[4/7] Загружается модель %s и переносится на GPU...", config["model_name"])
    stage_start = time.perf_counter()
    # FP32 parameters and optimizer states for both methods; BF16 is autocast only.
    dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        revision=config["model_revision"],
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to("cuda")
    logging.info("Модель готова за %.1f сек.", time.perf_counter() - stage_start)
    save_json(run_dir / "environment.json", {
        "python": sys.version,
        "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "datasets", "accelerate", "huggingface-hub"]},
        "parameter_dtype": str(dtype),
        "autocast_bf16": config["bf16"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "model_commit": getattr(model.config, "_commit_hash", None),
    })
    logging.info("[5/7] Оценка исходной модели на test. Обучение начнется после этого этапа.")
    base_metrics, base_predictions = evaluate_base(
        model, tokenizer, test_data, tokenized_test, config, run_dir,
    )
    # Baseline cache hit/miss must not affect training RNG state.
    set_seed(config["seed"])

    # LoRA: базовые веса заморожены; обучаются только адаптеры.
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["r"],
        lora_alpha=config["alpha"],
        lora_dropout=config["dropout"],
        target_modules=config["target_modules"],
        bias="none",
        use_rslora=False,
    ))

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    model.config.use_cache = False
    logging.info("Обучаемых параметров: %s из %s, доля: %s.", trainable_parameters, total_parameters, trainable_parameters / total_parameters * 100)
    arguments = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=config["bf16"],
        gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=config["logging_steps"],
        logging_first_step=True,
        disable_tqdm=False,
        report_to="none",
        skip_memory_metrics=True,
        seed=config["seed"],
        data_seed=config["seed"],
        label_names=["labels"],
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=tokenized_train,
        data_collator=collate,
        processing_class=tokenizer,
        callbacks=[TrainingLogger(run_dir)],
    )
    # Оставляем только нашу полоску с эпохой, loss и оставшимися шагами.
    trainer.remove_callback(ProgressCallback)
    trainer.remove_callback(PrinterCallback)
    logging.info("[6/7] Подготовлен Trainer. Начинается обучение...")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    resources = {
        "training_seconds": time.perf_counter() - start,
        "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 1024 ** 3,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_percent": 100 * trainable_parameters / total_parameters,
        "effective_batch_size": config["batch_size"] * config["gradient_accumulation_steps"],
    }
    save_json(run_dir / "resources.json", resources)
    # Сохраняем последнюю эпоху. Test не используется для выбора модели.
    logging.info("Обучение заняло %.1f сек. Сохраняется модель в %s...", resources["training_seconds"], run_dir / "model")
    model.config.use_cache = True
    trainer.save_model(str(run_dir / "model"))
    tokenizer.save_pretrained(run_dir / "model")
    logging.info("Модель и токенизатор сохранены.")
    del trainer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("[7/7] Итоговая оценка обученной модели на test...")
    tuned_metrics, tuned_predictions = evaluate(
        model, tokenizer, test_data, tokenized_test, config, run_dir, "tuned",
    )
    manual_examples = []
    for index in range(config["manual_examples"]):
        before = base_predictions[index]
        after = tuned_predictions[index]
        manual_examples.append({
            "test_id": index,
            "question": before["question"],
            "reference": before["reference"],
            "base_answer": before["prediction"],
            "tuned_answer": after["prediction"],
            "base_correct": before["correct"],
            "tuned_correct": after["correct"],
            "human_comment": "",
        })
    save_json(run_dir / "manual_review.json", manual_examples)
    result = {
        "protocol_id": fingerprint(json.loads((run_dir / "evaluation_protocol.json").read_text())),
        "dataset": json.loads((run_dir / "dataset.json").read_text()),
        "run": run_dir.name,
        "config": config,
        "base": base_metrics,
        "tuned": tuned_metrics,
        "resources": resources,
    }
    save_json(run_dir / "results.json", result)
    # Общая таблица: только завершенные запуски, одна строка на запуск.
    all_results = []
    for path in sorted(RUNS_DIR.glob("*/results.json")):
        status_path = path.parent / "status.json"
        status = json.loads(status_path.read_text())["status"] if status_path.exists() else None
        if path.parent == run_dir or status == "completed":
            with open(path, encoding="utf-8") as file:
                all_results.append(json.load(file))
    save_json(RUNS_DIR / "comparison.json", all_results)
    logging.info("Результаты сохранены: %s", run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sweep", action="store_true", help="Все девять конфигураций, каждая в новом процессе")
    group.add_argument("--experiment", choices=[item["name"] for item in LORA_EXPERIMENTS])
    args = parser.parse_args()
    if args.sweep:
        for changes in LORA_EXPERIMENTS:
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--experiment", changes["name"]], check=True)
    else:
        config = CONFIG.copy()
        if args.experiment:
            config.update(next(item for item in LORA_EXPERIMENTS if item["name"] == args.experiment))
        run_experiment(config)
