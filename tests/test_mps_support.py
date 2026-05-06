
import os
import torch
import unittest
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

class TestMPSSupport(unittest.TestCase):
    """
    Test script to verify Unsloth functionality on MPS.
    Note: This is intended to be run on actual Apple Silicon hardware.
    On other platforms, it will attempt to mock or skip device-specific checks.
    """

    @classmethod
    def setUpClass(cls):
        cls.model_name = "unsloth/Llama-3.2-1B-Instruct" # Small model for testing
        cls.max_seq_length = 512
        cls.dtype = None # Auto detection
        cls.load_in_4bit = False # Use 16-bit for baseline test

    def test_device_detection(self):
        from unsloth.device_type import DEVICE_TYPE, DEVICE_TYPE_TORCH
        print(f"\nDetected DEVICE_TYPE: {DEVICE_TYPE}")
        print(f"Detected DEVICE_TYPE_TORCH: {DEVICE_TYPE_TORCH}")

        if torch.backends.mps.is_available():
            self.assertEqual(DEVICE_TYPE, "mps")
            self.assertEqual(DEVICE_TYPE_TORCH, "mps")
        else:
            print("MPS not available on this system.")

    def test_model_loading(self):
        print("\nTesting model loading...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
        )
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)
        print(f"Model loaded on device: {model.device}")

    def test_sft_training_loop(self):
        print("\nTesting SFT training loop...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
        )

        dataset = Dataset.from_dict({
            "instruction": ["What is 2+2?", "Who is the CEO of Unsloth?"],
            "input": ["", ""],
            "output": ["2+2 is 4.", "Daniel Han and Michael Han are the founders."]
        })

        def formatting_prompts_func(examples):
            instructions = examples["instruction"]
            inputs       = examples["input"]
            outputs      = examples["output"]
            texts = []
            for instruction, input, output in zip(instructions, inputs, outputs):
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                texts.append(text)
            return { "text" : texts, }

        dataset = dataset.map(formatting_prompts_func, batched = True,)

        trainer = SFTTrainer(
            model = model,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = self.max_seq_length,
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 1,
                warmup_steps = 5,
                max_steps = 10,
                learning_rate = 2e-4,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 1,
                output_dir = "outputs",
                report_to = "none",
            ),
        )

        trainer_stats = trainer.train()
        self.assertIsNotNone(trainer_stats)
        print("Training completed successfully.")

    def test_inference(self):
        print("\nTesting inference...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
        )
        FastLanguageModel.for_inference(model)

        inputs = tokenizer(
            ["### Instruction:\nWhat is 2+2?\n\n### Response:\n"],
            return_tensors = "pt",
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens = 20)
        decoded = tokenizer.batch_decode(outputs)
        self.assertTrue(len(decoded) > 0)
        print(f"Generated text: {decoded[0]}")

if __name__ == "__main__":
    unittest.main()
