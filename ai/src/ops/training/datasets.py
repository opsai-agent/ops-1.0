"""Dataset preparation for OPS training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


class DatasetBuilder:
    """Build and prepare training datasets for OPS."""

    DATASETS = {
        "open_orca": {
            "name": "open-orca/platypus2",
            "description": "General instruction tuning dataset",
            "format": "alpaca"
        },
        "code_alpaca": {
            "name": "sahil2801/CodeAlpaca-20K",
            "description": "Code-focused instruction dataset",
            "format": "alpaca"
        },
        "soda": {
            "name": "bertdeclerck/soda",
            "description": "Conversational dataset",
            "format": "soda"
        },
        "math": {
            "name": "github/dolma:math",
            "description": "Math reasoning dataset",
            "format": "dolma"
        },
    }

    def __init__(self, output_dir: str = "./training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, dataset_name: str) -> Path:
        """Download a dataset."""
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset_info = self.DATASETS[dataset_name]
        logger.info(f"Downloading {dataset_info['name']}...")

        from datasets import load_dataset
        dataset = load_dataset(dataset_info["name"], split="train")

        # Save to local directory
        output_path = self.output_dir / f"{dataset_name}.jsonl"
        with open(output_path, "w") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")

        logger.info(f"Dataset saved to {output_path}")
        return output_path

    def convert_to_alpaca_format(
        self,
        input_path: Path,
        output_path: Path | None = None
    ) -> Path:
        """Convert dataset to Alpaca format for training."""
        if output_path is None:
            output_path = input_path.with_suffix(".alpaca.json")

        alpaca_data = []
        with open(input_path, "r") as f:
            for line in f:
                item = json.loads(line)
                # Try to extract instruction and output
                if "instruction" in item:
                    alpaca_data.append({
                        "instruction": item.get("instruction", ""),
                        "input": item.get("input", ""),
                        "output": item.get("output", ""),
                    })
                elif "conversations" in item:
                    # Convert Soda format to Alpaca
                    conversations = item["conversations"]
                    if len(conversations) >= 2:
                        alpaca_data.append({
                            "instruction": conversations[0].get("value", ""),
                            "input": "",
                            "output": conversations[1].get("value", ""),
                        })

        with open(output_path, "w") as f:
            json.dump(alpaca_data, f, indent=2)

        logger.info(f"Converted {len(alpaca_data)} samples to Alpaca format")
        return output_path

    def split_dataset(
        self,
        dataset_path: Path,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05
    ) -> dict[str, Path]:
        """Split dataset into train/val/test."""
        import random

        with open(dataset_path, "r") as f:
            data = [json.loads(line) for line in f]

        random.shuffle(data)

        train_size = int(len(data) * train_ratio)
        val_size = int(len(data) * val_ratio)

        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        splits = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }

        for split_name, split_data in splits.items():
            output_path = dataset_path.parent / f"{dataset_path.stem}_{split_name}.json"
            with open(output_path, "w") as f:
                json.dump(split_data, f, indent=2)
            logger.info(f"{split_name}: {len(split_data)} samples -> {output_path}")

        return {name: path.parent / f"{dataset_path.stem}_{name}.json"
                for name, path in zip(splits.keys(), [output_path] * 3)}

    def prepare_training_data(
        self,
        dataset_name: str,
        format: str = "alpaca"
    ) -> dict[str, Path]:
        """Prepare complete training dataset."""
        # Download
        raw_path = self.download_dataset(dataset_name)

        # Convert format
        if format == "alpaca":
            converted_path = self.convert_to_alpaca_format(raw_path)
        else:
            converted_path = raw_path

        # Split
        return self.split_dataset(converted_path)


def main():
    """CLI for dataset preparation."""
    import click

    @click.command()
    @click.option("--dataset", default="open_orca", help="Dataset to use")
    @click.option("--format", default="alpaca", help="Output format")
    @click.option("--output-dir", default="./training_data", help="Output directory")
    def prepare(dataset, format, output_dir):
        builder = DatasetBuilder(output_dir)
        paths = builder.prepare_training_data(dataset, format)
        logger.info("Dataset preparation complete!")
        for split, path in paths.items():
            logger.info(f"  {split}: {path}")

    prepare()


if __name__ == "__main__":
    main()
