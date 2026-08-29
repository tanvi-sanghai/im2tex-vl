"""Dataset loading, formatting, and demo-sample export utilities."""

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


DATASET_NAME = "unsloth/LaTeX_OCR"
INSTRUCTION = "Write the LaTeX representation for this image."
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIRECTORY = PROJECT_ROOT / "data" / "sample"


def load_raw_dataset() -> Dataset:
    """Load the training split used for fine-tuning."""
    return load_dataset(DATASET_NAME, split="train")


def convert_to_conversation(sample: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Convert one LaTeX OCR example to the Qwen2-VL conversation format."""
    instruction = "Write the LaTeX representation for this image."

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["text"]},
                ],
            },
        ]
    }


def export_demo_samples(dataset: Dataset, output_directory: Path = SAMPLE_DIRECTORY) -> None:
    """Save five image/LaTeX pairs for the README and evaluation demo."""
    output_directory.mkdir(parents=True, exist_ok=True)

    labels: dict[str, str] = {}
    for index, sample in enumerate(dataset.select(range(min(5, len(dataset)))), start=1):
        filename = f"sample_{index:02d}.png"
        sample["image"].convert("RGB").save(output_directory / filename, format="PNG")
        labels[filename] = sample["text"]

    with (output_directory / "labels.json").open("w", encoding="utf-8") as labels_file:
        json.dump(labels, labels_file, ensure_ascii=False, indent=2)
        labels_file.write("\n")


def load_and_format(n_train: int | None = None) -> Dataset:
    """Load the dataset and return it in the vision-language conversation format."""
    dataset = load_raw_dataset()
    if n_train is not None:
        dataset = dataset.select(range(min(n_train, len(dataset))))
    return dataset.map(convert_to_conversation)


def main() -> None:
    """Inspect the source fields and export the five committed demo samples."""
    dataset = load_raw_dataset()

    # The documented source fields are `image` and `text`, where `text` is LaTeX.
    print(dataset[0].keys())

    export_demo_samples(dataset)
    print(f"Saved five demo samples to {SAMPLE_DIRECTORY}")


if __name__ == "__main__":
    main()
