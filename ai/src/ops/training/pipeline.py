"""OPS Training Pipeline - Continue training on Meta LLaMA models."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Model settings
    base_model: str = "meta-llama/Meta-Llama-3-8B"
    output_dir: str = "./output"

    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Sequence length
    max_seq_length: int = 2048

    # Data settings
    dataset_name: str = "open-orca/platypus2"
    dataset_config: str = "default"
    val_set_size: float = 0.05

    # LoRA settings (parameter-efficient fine-tuning)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Quantization
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"

    # Logging
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3
    report_to: str = "none"

    # Evaluation
    evaluation_strategy: str = "steps"
    eval_steps: int = 200


class TrainingPipeline:
    """
    OPS Training Pipeline for custom-trained LLaMA models.

    Supports:
    - Supervised Fine-Tuning (SFT)
    - LoRA/QLoRA adaptation
    - RLHF (Reinforcement Learning from Human Feedback)
    - Multi-stage training
    """

    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check required dependencies are installed."""
        required = ["transformers", "torch", "peft", "trl"]
        missing = []
        for pkg in required:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            raise ImportError(f"Missing packages: {', '.join(missing)}\nInstall with: pip install {' '.join(missing)}")

    def load_dataset(self) -> Any:
        """Load training dataset from HuggingFace Hub or local file."""
        from datasets import load_dataset

        # Check if dataset_name is a local file path
        dataset_path = self.config.dataset_name
        if dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
            # Load local dataset
            if dataset_path.endswith('.jsonl'):
                dataset = load_dataset('json', data_files=dataset_path, split='train')
            else:
                dataset = load_dataset('json', data_files=dataset_path, split='train')
        else:
            # Load from HuggingFace Hub
            dataset = load_dataset(
                dataset_path,
                split="train",
                streaming=False,
            )

        if self.config.val_set_size > 0:
            dataset = dataset.train_test_split(test_size=self.config.val_set_size)
            train_dataset = dataset["train"]
            eval_dataset = dataset["test"]
        else:
            train_dataset = dataset
            eval_dataset = None

        return train_dataset, eval_dataset

    def prepare_dataset(
        self,
        dataset: Any,
        tokenizer: Any,
        max_length: int = 2048
    ) -> Any:
        """Prepare dataset for training."""
        def tokenize_fn(example):
            # Support both Alpaca format (instruction, input, output) and conversations format
            if "instruction" in example:
                # Alpaca format: convert to conversation
                instruction = example.get("instruction", "")
                input_text = example.get("input", "")
                output = example.get("output", "")

                if input_text:
                    user_content = f"{instruction}\n\n{input_text}"
                else:
                    user_content = instruction

                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output}
                ]
            elif "conversations" in example:
                # Already in conversations format
                messages = []
                for msg in example["conversations"]:
                    role = msg.get("from", "")
                    content = msg.get("value", "")
                    if role == "human":
                        messages.append({"role": "user", "content": content})
                    elif role == "gpt":
                        messages.append({"role": "assistant", "content": content})
            else:
                # Try to find text field
                text = example.get("text", "")
                if not text:
                    return {"input_ids": [], "attention_mask": [], "labels": []}
                messages = [{"role": "user", "content": text}]

            # Format as LLaMA chat
            text = self._format_conversation_from_messages(messages)
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            # Remove batch dimension
            result = {k: v.squeeze(0) for k, v in tokenized.items()}
            # For causal LM, labels are the same as input_ids
            result["labels"] = result["input_ids"].clone()
            return result

        return dataset.map(
            tokenize_fn,
            batched=False,
            remove_columns=dataset.column_names,
        )

    def _format_conversation(self, conversations: list[dict]) -> str:
        """Format conversation for LLaMA training (legacy format with 'from'/'value')."""
        formatted = ""
        for msg in conversations:
            role = msg.get("from", "")
            value = msg.get("value", "")
            if role == "human":
                formatted += f"<|start_header_id|>user<|end_header_id|>\n{value}<|eot_id|>\n"
            elif role == "gpt":
                formatted += f"<|start_header_id|>assistant<|end_header_id|>\n{value}<|eot_id|>\n"
        return formatted

    def _format_conversation_from_messages(self, messages: list[dict]) -> str:
        """Format conversation for LLaMA training from standard message format (role/content)."""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                formatted += f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>\n"
            elif role == "assistant":
                formatted += f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>\n"
            elif role == "system":
                formatted += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>\n"
        return formatted

    def train(self) -> None:
        """Run the training pipeline."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        import torch

        logger.info("Starting OPS training pipeline...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            padding_side="right",
        )
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

        # Load model
        if self.config.use_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=torch.bfloat16 if self.config.bnb_4bit_compute_dtype == "bfloat16" else torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.bfloat16 if self.config.bnb_4bit_compute_dtype == "bfloat16" else torch.float16,
                device_map="auto",
            )
        model = prepare_model_for_kbit_training(model)

        # Apply LoRA
        if self.config.use_lora:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        # Load and prepare dataset
        train_dataset, eval_dataset = self.load_dataset()
        train_dataset = self.prepare_dataset(train_dataset, tokenizer, self.config.max_seq_length)

        # Training arguments
        import torch
        use_cuda = torch.cuda.is_available()

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            report_to=self.config.report_to,
            eval_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            bf16=use_cuda,
            fp16=False,
            remove_unused_columns=False,
        )

        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        trainer.train()

        # Save model
        save_path = Path(self.config.output_dir) / "final_model"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        logger.info(f"Model saved to {save_path}")

    def evaluate(self, model_path: str) -> dict[str, float]:
        """Evaluate the trained model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Load evaluation dataset
        eval_dataset, _ = self.load_dataset()

        # Simple perplexity evaluation
        from torch.utils.data import DataLoader
        data_loader = DataLoader(eval_dataset, batch_size=1, collate_fn=lambda x: x)

        model.eval()
        total_loss = 0
        count = 0

        with torch.no_grad():
            for batch in data_loader:
                inputs = tokenizer(batch[0].get("text", ""), return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item()
                count += 1
                if count >= 100:  # Limit evaluation
                    break

        perplexity = torch.exp(torch.tensor(total_loss / count))
        return {"perplexity": perplexity.item(), "loss": total_loss / count}


def main():
    """CLI entry point for training."""
    import click

    @click.command()
    @click.option("--base-model", default="meta-llama/Meta-Llama-3-8B", help="Base model")
    @click.option("--dataset", default="open-orca/platypus2", help="Dataset name")
    @click.option("--epochs", default=3, help="Number of training epochs")
    @click.option("--output-dir", default="./output", help="Output directory")
    def train_command(base_model, dataset, epochs, output_dir):
        config = TrainingConfig(
            base_model=base_model,
            dataset_name=dataset,
            epochs=epochs,
            output_dir=output_dir,
        )
        pipeline = TrainingPipeline(config)
        pipeline.train()

    train_command()


if __name__ == "__main__":
    main()
