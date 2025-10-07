import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

try:
    import torch
except ImportError:  # torch is required for training; script will fail if not installed
    torch = None


@dataclass
class TrainConfig:
    base_model: str
    dataset_path: str
    output_dir: str
    lr: float = 2e-4
    batch_size: int = 2
    micro_batch_size: int = 2
    num_epochs: int = 1
    max_length: int = 512
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    use_4bit: bool = False
    gradient_accumulation_steps: Optional[int] = None


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return TrainConfig(**cfg)


def build_prompt(example: dict) -> str:
    if "messages" in example:
        # simple chat -> single turn concat
        lines = []
        for m in example["messages"]:
            role = m.get("role", "user").capitalize()
            lines.append(f"{role}: {m.get('content','')}")
        if not lines[-1].startswith("Assistant:"):
            lines.append("Assistant:")
        return "\n".join(lines)
    instr = example.get("instruction", "")
    inp = example.get("input", "").strip()
    out = example.get("output", "")
    header = f"System: You are a helpful assistant.\nUser: {instr}\n"
    if inp:
        header += f"Context: {inp}\n"
    header += "Assistant: "
    return header + out


def main() -> None:
    ap = argparse.ArgumentParser(description="Config-driven LoRA fine-tuning")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        device_map="auto",
        load_in_4bit=cfg.use_4bit,
    )
    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_cfg)

    ds = load_dataset("json", data_files=cfg.dataset_path)["train"]

    def tokenize_fn(batch):
        texts = [build_prompt(ex) for ex in batch]
        toks = tokenizer(
            texts,
            max_length=cfg.max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        toks["labels"] = toks["input_ids"].copy()
        return toks

    tokenized = ds.map(tokenize_fn, batched=False, remove_columns=ds.column_names)

    grad_accum = cfg.gradient_accumulation_steps or max(
        1, cfg.batch_size // cfg.micro_batch_size
    )

    use_bf16 = False
    use_fp16 = False
    precision_warning = None

    if torch is None:
        precision_warning = (
            "PyTorch is not available; training will proceed in full precision."
        )
    else:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            bf16_supported = False
            is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
            if callable(is_bf16_supported):
                bf16_supported = is_bf16_supported()
            if bf16_supported:
                use_bf16 = True
            else:
                use_fp16 = True
                precision_warning = (
                    "CUDA device does not support bfloat16; falling back to float16 precision."
                )
        else:
            precision_warning = (
                "CUDA is not available; training will proceed in full precision."
            )

    if precision_warning:
        warnings.warn(precision_warning)

    args_tr = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=cfg.num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=[],
    )

    class CsvLogger(TrainerCallback):
        def __init__(self, path: Path):
            self.path = path
            self._fh = path.open("w", encoding="utf-8")
            self._fh.write("step,loss\n")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self._fh.write(f"{int(state.global_step)},{logs['loss']}\n")
                self._fh.flush()

        def on_train_end(self, args, state, control, **kwargs):
            try:
                self._fh.close()
            except Exception:
                pass

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.add_callback(CsvLogger(out_dir / "train_logs.csv"))
    trainer.train()

    # Save adapter-only by default
    model.save_pretrained(str(out_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(out_dir))

    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump({"params": vars(cfg)}, f, indent=2)


if __name__ == "__main__":
    main()
