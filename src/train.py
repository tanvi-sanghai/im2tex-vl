"""Fine-tune Qwen2-VL-2B-Instruct with LoRA in Google Colab."""

from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTConfig, SFTTrainer

from src.data import load_and_format


def main() -> None:
    """Run the Colab fine-tuning job and save its LoRA adapter."""
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen2-VL-2B-Instruct",  # 2B, not 7B — fits comfortably on a free T4
        load_in_4bit=True,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    converted_dataset = load_and_format()

    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=100,  # Official demo uses 30; use 100 for this project's time budget.
            learning_rate=2e-4,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=2048,
        ),
    )

    trainer_stats = trainer.train()
    print(trainer_stats)

    model.save_pretrained("checkpoints/lora_adapter")
    tokenizer.save_pretrained("checkpoints/lora_adapter")


if __name__ == "__main__":
    main()
