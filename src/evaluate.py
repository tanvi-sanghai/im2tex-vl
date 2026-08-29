"""Compare base and LoRA-adapted Qwen2-VL predictions in Google Colab."""

import json
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from unsloth import FastVisionModel

from src.data import INSTRUCTION


MODEL_NAME = "unsloth/Qwen2-VL-2B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIRECTORY = PROJECT_ROOT / "data" / "sample"
ADAPTER_DIRECTORY = PROJECT_ROOT / "checkpoints" / "lora_adapter"
COMPARISONS_PATH = PROJECT_ROOT / "results" / "comparisons.md"


def load_model(adapter_directory: Path | None = None):
    """Load the base Qwen2-VL model, optionally with the saved LoRA adapter."""
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
    )
    if adapter_directory is not None:
        model = PeftModel.from_pretrained(model, adapter_directory)

    FastVisionModel.for_inference(model)
    return model, tokenizer


def predict(model, tokenizer, image: Image.Image) -> str:
    """Generate LaTeX for one equation image using the training instruction."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": INSTRUCTION},
                {"type": "image", "image": image},
            ],
        }
    ]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        use_cache=True,
        do_sample=False,
    )
    response_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    return tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0].strip()


def predict_samples(model, tokenizer, filenames: list[str]) -> dict[str, str]:
    """Run inference for each committed demo image."""
    predictions: dict[str, str] = {}
    for filename in filenames:
        with Image.open(SAMPLE_DIRECTORY / filename) as image:
            predictions[filename] = predict(model, tokenizer, image.convert("RGB"))
    return predictions


def escape_markdown_cell(value: str) -> str:
    """Keep LaTeX values intact and safe inside a Markdown table cell."""
    return value.replace("|", "\\|").replace("\n", "<br>")


def write_comparisons(
    labels: dict[str, str],
    base_predictions: dict[str, str],
    fine_tuned_predictions: dict[str, str],
) -> None:
    """Write the before-and-after results table used by the README."""
    lines = [
        "# Base vs. Fine-Tuned Qwen2-VL",
        "",
        "| Image filename | Ground-truth LaTeX | Base model prediction | Fine-tuned model prediction |",
        "| --- | --- | --- | --- |",
    ]
    for filename, ground_truth in labels.items():
        lines.append(
            "| {filename} | {ground_truth} | {base} | {fine_tuned} |".format(
                filename=filename,
                ground_truth=escape_markdown_cell(ground_truth),
                base=escape_markdown_cell(base_predictions[filename]),
                fine_tuned=escape_markdown_cell(fine_tuned_predictions[filename]),
            )
        )

    COMPARISONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPARISONS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Evaluate both models on the committed samples after Colab training."""
    if not torch.cuda.is_available():
        raise RuntimeError("Run evaluation in the same GPU-enabled Colab session as training.")
    if not ADAPTER_DIRECTORY.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {ADAPTER_DIRECTORY}")

    labels = json.loads((SAMPLE_DIRECTORY / "labels.json").read_text(encoding="utf-8"))
    filenames = list(labels)

    base_model, tokenizer = load_model()
    base_predictions = predict_samples(base_model, tokenizer, filenames)
    del base_model
    torch.cuda.empty_cache()

    fine_tuned_model, tokenizer = load_model(ADAPTER_DIRECTORY)
    fine_tuned_predictions = predict_samples(fine_tuned_model, tokenizer, filenames)
    write_comparisons(labels, base_predictions, fine_tuned_predictions)

    print(f"Wrote comparison table to {COMPARISONS_PATH}")


if __name__ == "__main__":
    main()
