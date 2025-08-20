import argparse
import os
import numpy as np
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score

from utils import set_seed, device_str, get_label_names_from_dataset


try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


@dataclass
class Config:
    model_name: str = "distilbert-base-uncased"
    output_dir: str = "models/distilbert-banking77"
    seed: int = 42
    batch_size: int = 16
    lr: float = 5e-5
    num_epochs: int = 5
    weight_decay: float = 0.01
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1


def tokenize_fn(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=Config.model_name)
    parser.add_argument("--output_dir", type=str, default=Config.output_dir)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--num_epochs", type=int, default=Config.num_epochs)
    parser.add_argument("--weight_decay", type=float, default=Config.weight_decay)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=Config.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=Config.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=Config.lora_dropout)
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading dataset…")
    ds = load_dataset("banking77")
    split = ds["train"].train_test_split(test_size=0.1, seed=args.seed)
    train_ds = split["train"]
    val_ds = split["test"]
    test_ds = ds["test"]

    label_names = get_label_names_from_dataset(ds)
    num_labels = len(label_names)

    print(f"Loading tokenizer & model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels
    )

    # Optional: LoRA fine-tuning
    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("PEFT is not installed. Install `peft` or run without --use_lora.")
        target_modules = ["q_lin", "v_lin"]
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    print("Tokenizing…")
    encoded_train = train_ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True)
    encoded_val = val_ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True)
    encoded_test = test_ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)

    fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",  
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        logging_steps=50,
        fp16=fp16,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        eval_dataset=encoded_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"Training on {device_str()}…")
    trainer.train()

    print("Evaluating on validation set…")
    val_metrics = trainer.evaluate(eval_dataset=encoded_val)
    print(val_metrics)

    print("Evaluating on test set…")
    test_metrics = trainer.evaluate(eval_dataset=encoded_test)
    print(test_metrics)

    
    import json
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "val_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in val_metrics.items()}, f, indent=2)
    with open(os.path.join("reports", "test_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)

    
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "label_names.txt"), "w") as f:
        for name in label_names:
            f.write(name + "\n")

    print("Done. Best model + tokenizer saved to:", args.output_dir)


if __name__ == "__main__":
    main()