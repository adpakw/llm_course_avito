import glob
import math
import os
from datasets import Dataset, load_dataset
import plotly.graph_objects as go
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)
import torch
import time


# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 15
MAX_LENGTH = 512
INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
LABELS = 'labels'

# Don't change these parameters
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000


# TODO: Configure training parameters
TRAINING_CONFIG = {
    'output_dir': f'{OUTPUT_DIR}/gpt2-1b-russian',
    'optim': 'adamw_torch_fused',
    'num_train_epochs': 1,
    'per_device_train_batch_size': 16,
    'save_steps': 100,
    'save_total_limit': 2,
    'learning_rate': 3e-4,
    'weight_decay': 0.01,
    'warmup_steps': 200,
    'logging_steps': 1,
    'eval_steps': 100,
    'eval_strategy': 'steps',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    'bf16': True,
    'tf32': True,
    'gradient_checkpointing': False,
    'gradient_accumulation_steps': 1,
    'dataloader_num_workers': 4,
    'torch_compile': False,
    'report_to': 'none',
}


class TimeoutCallback(TrainerCallback):
    """Callback to stop training after a specified timeout."""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                control.should_training_stop = True
                # Include the final weights in best-checkpoint selection.
                control.should_evaluate = True
                control.should_save = True
                print(f"Training stopped after {elapsed:.2f} seconds")
        return control


def prepare_tokenizer():
    """
    TODO: Implement tokenizer preparation.
    - Load the tokenizer from TOKENIZER_NAME
    - Set pad_token to eos_token
    - Return the tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def tokenize_function(examples, tokenizer):
    """
    TODO: Implement tokenization function.
    - Tokenize the text with truncation and padding to MAX_LENGTH
    - Create labels from input_ids
    - Return dictionary with 'labels', 'input_ids', and 'attention_mask'
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors=None,
    )
    tokenized[LABELS] = [
        [
            token if token != tokenizer.pad_token_id else -100
            for token in seq
        ]
        for seq in tokenized[INPUT_IDS]
    ]
    return tokenized


def save_as_parquets(ds, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    """
    TODO: Implement saving dataset as parquet shards.
    - Create output directory if it doesn't exist
    - Split dataset into num_shards shards
    - Save each shard as a parquet file with format: {output_dir}/{index:05d}.parquet
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Saving dataset")
    for i in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=i, contiguous=True)
        shard_path = os.path.join(output_dir, f"{i:05d}.parquet")
        shard.to_parquet(shard_path)
    print("Dataset successfully saved")


def prepare_dataset():
    """
    TODO: Implement dataset preparation.
    - Load the Wikipedia dataset: "wikimedia/wikipedia", "20231101.ru", split="train"
    - Tokenize the dataset using tokenize_function
    - Save as parquet files
    """
    print("Loading dataset")
    raw_dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")
    tokenizer = prepare_tokenizer()

    print("Tokenizing dataset")
    tokenized_dataset = raw_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        batch_size=2056,
        num_proc=os.cpu_count(),
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing dataset",
    )
    save_as_parquets(tokenized_dataset, OUTPUT_DIR, NUM_SHARDS)
    print("Dataset tokenized")



def load_tokenized_dataset(data_dir=OUTPUT_DIR):
    """
    TODO: Implement loading of tokenized dataset from parquet files.
    - List only parquet files in data_dir, sorted by filename
    - Load them from local parquet files
    - Return the 'train' split
    """
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError("No tokenized parquet files found")
    print(f"Loading tokenized dataset from {len(parquet_files)} parquet files in {data_dir}")
    return Dataset.from_parquet(
        parquet_files,
        cache_dir=os.path.join(data_dir, ".parquet_cache"),
    )


def plot_loss_history(log_history, output_dir):
    """
    Save train/eval loss history as a Plotly HTML chart.
    """
    os.makedirs(output_dir, exist_ok=True)

    train_points = [
        (entry["step"], entry["loss"])
        for entry in log_history
        if "step" in entry and "loss" in entry
    ]
    eval_points = [
        (entry["step"], entry["eval_loss"])
        for entry in log_history
        if "step" in entry and "eval_loss" in entry
    ]

    if not train_points and not eval_points:
        print("No loss values found in trainer logs. Loss plot was not created.")
        return

    fig = go.Figure()
    if train_points:
        train_steps, train_losses = zip(*train_points)
        fig.add_trace(
            go.Scatter(
                x=train_steps,
                y=train_losses,
                mode="lines",
                name="train loss",
            )
        )
    if eval_points:
        eval_steps, eval_losses = zip(*eval_points)
        fig.add_trace(
            go.Scatter(
                x=eval_steps,
                y=eval_losses,
                mode="lines+markers",
                name="eval loss",
            )
        )

    fig.update_layout(
        title="Training and evaluation loss",
        xaxis_title="Step",
        yaxis_title="Loss",
        template="plotly_white",
        hovermode="x unified",
        width=1000,
        height=600,
    )

    html_path = os.path.join(output_dir, "loss_curve.html")
    fig.write_html(html_path, include_plotlyjs="cdn")

    print(f"Loss Plotly HTML saved to {html_path}")


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    dataset_size = len(dataset)
    train_dataset = dataset.select(range(validation_size, dataset_size))
    eval_dataset = dataset.select(range(validation_size))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def create_model(tokenizer):
    # Don't change this parameter
    MODEL_CONFIG = {
        'hidden_size': 2048,
        'num_hidden_layers': 12,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 8192,
        'head_dim': 128,
        'hidden_act': 'silu',
        'initializer_range': 0.02,
        'scale_attn_weights': True,
        'use_cache': True,
    }

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG
    )
    
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16
    )
    
    print(f"Model pad token id: {model.config.pad_token_id}")
    
    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")
    
    return model


def generate_samples(
    model, 
    tokenizer, 
    prompts=None, 
    max_new_tokens=64, 
    device="cuda"
):
    if prompts is None:
        prompts = [
            "Столица Франции - ",
            "LLM работают на архитектуре ",
            "Самая большая страна в мире ",
            "Первый закон Ньютона гласит ",
        ]

    model.eval()

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=1,
                repetition_penalty=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Generation: {decoded}\n" + "-" * 50)


def train_model():
    """
    TODO: Implement the training pipeline.
    - Prepare tokenizer
    - Load tokenized dataset and split it
    - Create the model
    - Create TrainingArguments from TRAINING_CONFIG
    - Create Trainer with TimeoutCallback
    - Train the model
    - Run final evaluation and print results
    - Save metric history to trainer_state.json for local loss plots
    """
    tokenizer = prepare_tokenizer()
    full_dataset = load_tokenized_dataset(OUTPUT_DIR)
    train_dataset, eval_dataset = split_dataset(full_dataset, VALIDATION_SIZE)

    model = create_model(tokenizer)

    training_args = TrainingArguments(**TRAINING_CONFIG)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=[TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS)] # dont change
    )

    trainer.train()

    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss", float("nan"))
    perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
    print(f"Final Eval Loss: {eval_loss:.4f} | Perplexity: {perplexity:.2f}")

    trainer.save_state()
    plot_loss_history(trainer.state.log_history, TRAINING_CONFIG["output_dir"])
    trainer.save_model(os.path.join(TRAINING_CONFIG["output_dir"], "best_model"))

    generate_samples(model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Step 1: Prepare the dataset (run once)
    if not os.path.exists(OUTPUT_DIR) or not glob.glob(
        os.path.join(OUTPUT_DIR, "*.parquet")
    ):
        prepare_dataset()
    
    # Step 2: Train the model
    train_model()
