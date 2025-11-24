import argparse
import os

import pandas as pd
import torch
import wandb
from datasets import Dataset, DatasetDict
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DistilBERT on IMDb")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to directory containing train.parquet and test.parquet files",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size per device (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for model checkpoints (default: ./results)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased",
        help="Pretrained model name (default: distilbert-base-uncased)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)",
    )
    return parser.parse_args()


def load_data(data_dir):
    """Load dataset from local parquet files"""
    print(f"Loading data from {data_dir}...")
    
    train_path = os.path.join(data_dir, "train.parquet")
    test_path = os.path.join(data_dir, "test.parquet")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    # Load parquet files
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Convert to HuggingFace Dataset
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df),
    })
    
    print(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
    return dataset


def preprocess_data(dataset, tokenizer, max_length):
    """Tokenize the dataset"""
    print("Tokenizing dataset...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def compute_metrics(eval_pred):
    """Compute accuracy metric"""
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Training Configuration:")
    print(f"  Data Dir: {args.data_dir}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Model: {args.model_name}")
    print(f"  Output Dir: {args.output_dir}")
    print("=" * 60)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load dataset
    dataset = load_data(args.data_dir)
    
    # Load tokenizer and model
    print(f"\nLoading model: {args.model_name}...")
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    # Move model to GPU
    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"Model device: {next(model.parameters()).device}")
    
    # Preprocess data
    tokenized_dataset = preprocess_data(dataset, tokenizer, args.max_length)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        report_to="wandb",
        fp16=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Evaluate
    print("\nEvaluating model...")
    results = trainer.evaluate()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final Accuracy: {results['eval_accuracy']:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60)
    
    # Save the final model
    model.save_pretrained(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    print(f"\nFinal model saved to: {args.output_dir}/final_model")


if __name__ == "__main__":
    main()
