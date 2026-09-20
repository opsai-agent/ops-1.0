"""OPS Training entry point."""

from .pipeline import TrainingConfig, TrainingPipeline
from .datasets import DatasetBuilder

__all__ = ["TrainingPipeline", "TrainingConfig", "DatasetBuilder"]

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
