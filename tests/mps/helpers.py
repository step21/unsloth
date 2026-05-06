"""Shared scaffold for Layer 3 MPS quantization tests."""

import torch

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
MAX_SEQ = 512
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_RANK = 8


def _tiny_dataset():
    from datasets import Dataset
    return Dataset.from_dict({"text": [
        "### Instruction:\nWhat is 2+2?\n\n### Response:\n4",
        "### Instruction:\nWhat is the capital of France?\n\n### Response:\nParis",
        "### Instruction:\nWhat color is the sky?\n\n### Response:\nBlue",
        "### Instruction:\nWhat is 3+3?\n\n### Response:\n6",
        "### Instruction:\nName a planet.\n\n### Response:\nMars",
        "### Instruction:\nWhat is water made of?\n\n### Response:\nHydrogen and oxygen",
    ]})


def _apply_lora(model):
    from unsloth import FastLanguageModel
    return FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=LORA_TARGETS,
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        bias="none",
    )


def _mps_optimizer():
    """
    Return optimizer_cls_and_kwargs for MPS.
    Unsloth defaults to bnb adamw_8bit which does not support MPS.
    Prefer torchao AdamW8bit (MPS-native), fall back to standard adamw_torch.
    """
    try:
        import torchao.optim as tao
        return (tao.AdamW8bit, {"lr": 2e-4})
    except ImportError:
        from torch.optim import AdamW
        return (AdamW, {"lr": 2e-4})


def _run_training(model, tokenizer, fp16=True, bf16=False, output_dir="outputs"):
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model=model,
        train_dataset=_tiny_dataset(),
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=5,
            learning_rate=2e-4,
            fp16=fp16,
            bf16=bf16,
            logging_steps=1,
            report_to="none",
            dataset_text_field="text",
            save_strategy="no",
        ),
        optimizer_cls_and_kwargs=_mps_optimizer(),
    )
    trainer.train()
    return [x["loss"] for x in trainer.state.log_history if "loss" in x]


def assert_losses_finite(losses):
    assert len(losses) > 0, "No loss values were logged"
    assert all(torch.isfinite(torch.tensor(l)) for l in losses), (
        f"Non-finite loss detected: {losses}"
    )
