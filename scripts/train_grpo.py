"""Train an instruction therapist policy with GRPO using a local reward model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT_DIR / "reward_data" / "reward_training_data.jsonl"
DEFAULT_LLAMA_OUTPUT_DIR = ROOT_DIR / "grpo_models" / "llama_3_1_8b_rm"
DEFAULT_QWEN_OUTPUT_DIR = ROOT_DIR / "grpo_models" / "qwen_2_5_7b_1m_rm"
LLAMA_POLICY_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
QWEN_POLICY_MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"
POLICY_PRESETS = {
    "llama": {
        "model_name": LLAMA_POLICY_MODEL,
        "output_dir": DEFAULT_LLAMA_OUTPUT_DIR,
    },
    "qwen": {
        "model_name": QWEN_POLICY_MODEL,
        "output_dir": DEFAULT_QWEN_OUTPUT_DIR,
    },
}
DEFAULT_GRPO_PROMPT = ROOT_DIR / "prompts" / "aim_therapist" / "grpo_therapist.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Llama-3.1-8B-Instruct or Qwen2.5-7B-Instruct-1M with TRL "
            "GRPO using the local reward model/ranker trained from AIM reward data."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Reward JSONL data used to build therapy-context prompts.",
    )
    parser.add_argument(
        "--reward-model",
        type=Path,
        required=True,
        help="Local reward model/ranker checkpoint, e.g. reward_ranker_d1/final.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory where the GRPO-trained policy is saved. Defaults to a "
            "model-specific directory under grpo_models/."
        ),
    )
    parser.add_argument(
        "--policy-preset",
        choices=tuple(POLICY_PRESETS),
        default="llama",
        help=(
            "Base policy preset. Use 'qwen' for "
            f"{QWEN_POLICY_MODEL}. Defaults to llama."
        ),
    )
    parser.add_argument(
        "--model-name",
        help=(
            "Policy model to fine-tune. Overrides --policy-preset when set. "
            f"Examples: {LLAMA_POLICY_MODEL}, {QWEN_POLICY_MODEL}."
        ),
    )
    parser.add_argument(
        "--text-field",
        default="reward_model_input",
        help="JSONL field containing context plus original candidate response.",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=DEFAULT_GRPO_PROMPT,
        help=(
            "Prompt template for GRPO therapist generation. Must include "
            "{conversation_history}."
        ),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        help="Optional limit for quick experiments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=1024,
        help="Maximum prompt length for GRPO.",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=160,
        help="Maximum generated therapist response length.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Number of completions sampled per prompt group.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
        help="Number of GRPO training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-7,
        help="GRPO fine-tuning learning rate.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Logging interval.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Checkpoint save interval.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 mixed precision.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 mixed precision.",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Train with LoRA adapters instead of full fine-tuning.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the policy in 4-bit. Requires bitsandbytes.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the policy model.",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Enable TRL/vLLM generation acceleration if installed/configured.",
    )
    parser.add_argument(
        "--vllm-mode",
        choices=("colocate", "server"),
        default="colocate",
        help="TRL vLLM mode when --use-vllm is enabled.",
    )
    return parser.parse_args()


def resolve_policy_config(args: argparse.Namespace) -> None:
    preset = POLICY_PRESETS[args.policy_preset]
    if args.model_name is None:
        args.model_name = preset["model_name"]
    if args.output_dir is None:
        args.output_dir = preset["output_dir"]


def import_grpo_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing GRPO dependency. Install the RL dependencies, for example:\n"
            "pip install trl datasets peft accelerate transformers torch"
        ) from exc
    return torch, Dataset, AutoModelForSequenceClassification, AutoTokenizer, GRPOConfig, GRPOTrainer


def import_lora_config() -> Any:
    try:
        from peft import LoraConfig
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: peft. Install it with `pip install peft`.") from exc
    return LoraConfig


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_number}: {path}") from exc
            records.append(record)
    if not records:
        raise SystemExit(f"No records found in {path}")
    return records


def remove_final_therapist_utterance(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and lines[-1].startswith("Therapist:"):
        lines = lines[:-1]
    return "\n".join(lines).strip() or "No previous turns."


def load_prompt_template(path: Path) -> str:
    try:
        template = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing GRPO prompt template: {path}") from exc
    if "{conversation_history}" not in template:
        raise SystemExit(
            f"GRPO prompt template must include {{conversation_history}}: {path}"
        )
    return template


def prompt_from_reward_input(text: str, template: str) -> str:
    context = remove_final_therapist_utterance(text)
    return template.replace("{conversation_history}", context).strip()


def build_prompt_dataset(
    records: list[dict[str, Any]],
    text_field: str,
    prompt_template: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_prompts: set[str] = set()
    for record in records:
        text = record.get(text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        prompt = prompt_from_reward_input(text, prompt_template)
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        rows.append({"prompt": prompt})
    if not rows:
        raise SystemExit(f"No usable prompts found in field {text_field!r}.")
    return rows


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(completion)


def strip_prompt_prefix(prompt: str) -> str:
    marker = "Conversation so far:\n"
    if marker not in prompt:
        return prompt
    return prompt.split(marker, 1)[1].rsplit("\n\nTherapist response:", 1)[0].strip()


def make_reward_function(
    *,
    torch: Any,
    reward_model: Any,
    reward_tokenizer: Any,
    reward_device: Any,
    max_length: int,
) -> Any:
    def reward_func(prompts: list[Any], completions: list[Any], **_: Any) -> list[float]:
        texts = []
        for prompt, completion in zip(prompts, completions, strict=True):
            prompt_text = completion_to_text(prompt)
            context = strip_prompt_prefix(prompt_text)
            completion_text = completion_to_text(completion).strip()
            texts.append(f"{context}\nTherapist: {completion_text}".strip())

        encoded = reward_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(reward_device) for name, tensor in encoded.items()}
        with torch.inference_mode():
            scores = reward_model(**encoded).logits.reshape(-1)
        return [float(score) for score in scores.cpu().tolist()]

    return reward_func


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    resolve_policy_config(args)
    if args.max_records is not None and args.max_records < 1:
        raise SystemExit("--max-records must be at least 1.")
    if args.reward_model is not None and not args.reward_model.exists():
        raise SystemExit(f"Reward model checkpoint does not exist: {args.reward_model}")

    torch, Dataset, AutoModelForSequenceClassification, AutoTokenizer, GRPOConfig, GRPOTrainer = (
        import_grpo_dependencies()
    )

    prompt_template = load_prompt_template(args.prompt)
    records = read_jsonl(args.data)
    if args.max_records is not None:
        records = records[: args.max_records]
    rows = build_prompt_dataset(records, args.text_field, prompt_template)
    train_dataset = Dataset.from_list(rows)

    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model)
    reward_tokenizer.truncation_side = "left"
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model)
    reward_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model.to(reward_device)
    reward_model.eval()

    model_init_kwargs: dict[str, Any] = {}
    if args.trust_remote_code:
        model_init_kwargs["trust_remote_code"] = True
    if args.load_in_4bit:
        model_init_kwargs["load_in_4bit"] = True
        model_init_kwargs["device_map"] = "auto"

    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        seed=args.seed,
        model_init_kwargs=model_init_kwargs or None,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
    )

    peft_config = None
    if args.use_lora:
        LoraConfig = import_lora_config()
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    reward_func = make_reward_function(
        torch=torch,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        reward_device=reward_device,
        max_length=args.max_prompt_length + args.max_completion_length,
    )

    trainer_kwargs: dict[str, Any] = {
        "model": args.model_name,
        "reward_funcs": reward_func,
        "args": training_args,
        "train_dataset": train_dataset,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = GRPOTrainer(**trainer_kwargs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_dir / "grpo_config_summary.json",
        {
            "data": str(args.data),
            "reward_model": str(args.reward_model),
            "policy_preset": args.policy_preset,
            "model_name": args.model_name,
            "text_field": args.text_field,
            "prompt": str(args.prompt),
            "prompt_count": len(rows),
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "num_generations": args.num_generations,
            "use_lora": args.use_lora,
            "load_in_4bit": args.load_in_4bit,
            "trust_remote_code": args.trust_remote_code,
            "use_vllm": args.use_vllm,
            "vllm_mode": args.vllm_mode,
        },
    )
    trainer.train()
    trainer.save_model(args.output_dir / "final")
    print(f"Saved GRPO policy to {args.output_dir / 'final'}")


if __name__ == "__main__":
    main()
