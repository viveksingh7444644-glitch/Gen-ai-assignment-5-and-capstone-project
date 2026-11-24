"""Lab 3 - Text Generation Using GPT-2.

ETMMGA304 Laboratory Manual, pages 10-12.

1. Compares greedy decoding with nucleus sampling (temperature 0.6 / 1.0,
   top_p 0.9, seeds 42 / 43) on two robot-story prompts.
2. Freezes GPT-2 and fine-tunes only the last Transformer block on six
   study-tip examples (prompt tokens masked with -100).
3. Compares base and tuned models on held-out topics.

Outputs:
    lab_outputs/study_tip_gpt2/          (fine-tuned model + tokenizer)
    lab_outputs/lab3_decoding.csv        (all decoding outputs to score)
    lab_outputs/lab3_held_out.csv        (base vs tuned continuations)

Run:
    python lab3_gpt2_text_generation.py [--skip-decoding] [--epochs 3]
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedModel, PreTrainedTokenizerBase)

SEED = 42
OUT_DIR = Path("lab_outputs")
DATA_DIR = Path("data")
MODEL_ID = "openai-community/gpt2"
TUNED_DIR = OUT_DIR / "study_tip_gpt2"

PROMPTS: list[str] = [
    "A story about a robot begins like this:",
    "In a school library, a tiny robot found a lost book. It",
]
TEMPERATURES: tuple[float, ...] = (0.6, 1.0)
SEEDS: tuple[int, ...] = (42, 43)

# Six demonstration examples from the manual. Extend with varied,
# instructor-approved pairs for the capstone; never include held-out topics.
EXAMPLES: list[tuple[str, str]] = [
    ("graphs", "Sketch a graph, then mark a path through three vertices."),
    ("Python loops", "Trace a loop by writing the counter after each step."),
    ("fractions", "Draw equal parts and label the fraction you shaded."),
    ("sets", "Draw a Venn diagram and label the intersection."),
    ("sorting", "Sort five values by hand and record each swap."),
    ("probability", "List the outcomes of two coin tosses before counting."),
]
HELD_OUT: list[str] = ["recursion", "matrices", "binary search"]


def setup(seed: int = SEED) -> torch.device:
    """Seed RNGs, create output folders and return the compute device."""
    random.seed(seed)
    torch.manual_seed(seed)
    OUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {seed}")
    return device


def study_prompt(topic: str) -> str:
    return f"Topic: {topic}\nStudy tip:"


# --------------------------------------------------------------------------- #
# Step 1 - Load model
# --------------------------------------------------------------------------- #
def load_base(
    device: torch.device,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return tokenizer, model


# --------------------------------------------------------------------------- #
# Step 2 - Decoding comparison
# --------------------------------------------------------------------------- #
def continue_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    sample: bool = False,
    temperature: float = 0.8,
    seed: int = SEED,
    max_new_tokens: int = 45,
) -> str:
    """Generate a continuation of ``prompt`` (greedy unless ``sample``)."""
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    options: dict[str, object] = {
        "do_sample": sample,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if sample:
        options.update(temperature=temperature, top_p=0.9)
    with torch.no_grad():
        tokens = model.generate(**inputs, **options)
    new_tokens = tokens[0, inputs.input_ids.size(1):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def compare_decoding(model: PreTrainedModel,
                     tokenizer: PreTrainedTokenizerBase) -> list[dict]:
    rows: list[dict] = []
    for prompt in PROMPTS:
        print("\nPROMPT:", prompt)
        text = continue_text(model, tokenizer, prompt)
        print("GREEDY:", text)
        rows.append({"prompt": prompt, "decoding": "greedy", "seed": "",
                     "temperature": "", "output": text})
        for temp in TEMPERATURES:
            for seed in SEEDS:
                text = continue_text(model, tokenizer, prompt, sample=True,
                                     temperature=temp, seed=seed)
                print(f"T={temp} seed={seed}:", text)
                rows.append({"prompt": prompt, "decoding": "sample",
                             "seed": seed, "temperature": temp,
                             "output": text})
    return rows


# --------------------------------------------------------------------------- #
# Step 3 - Fine-tune the last Transformer block
# --------------------------------------------------------------------------- #
def fine_tune(tokenizer: PreTrainedTokenizerBase, device: torch.device,
              epochs: int = 3, lr: float = 2e-5) -> PreTrainedModel:
    tuned = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    tuned.requires_grad_(False)
    last_block = tuned.transformer.h[-1]
    last_block.requires_grad_(True)
    trainable = sum(p.numel() for p in tuned.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {trainable:,}")

    opt = torch.optim.AdamW(last_block.parameters(), lr=lr)
    for epoch in range(epochs):
        tuned.train()
        loss = torch.tensor(0.0)
        for topic, answer in EXAMPLES:
            prefix = tokenizer.encode(study_prompt(topic))
            target = tokenizer.encode(" " + answer) + [tokenizer.eos_token_id]
            ids = torch.tensor([prefix + target], device=device)
            # -100 masks prompt tokens: only the answer contributes to loss.
            labels = torch.tensor([[-100] * len(prefix) + target],
                                  device=device)
            opt.zero_grad()
            loss = tuned(input_ids=ids, labels=labels).loss
            loss.backward()
            opt.step()
        print(f"Epoch: {epoch + 1} last example loss: {loss.item():.3f}")

    tuned.eval()
    tuned.save_pretrained(TUNED_DIR)
    tokenizer.save_pretrained(TUNED_DIR)
    print(f"Saved fine-tuned model -> {TUNED_DIR}")
    return tuned


# --------------------------------------------------------------------------- #
# Step 4 - Held-out comparison
# --------------------------------------------------------------------------- #
def compare_held_out(base: PreTrainedModel, tuned: PreTrainedModel,
                     tokenizer: PreTrainedTokenizerBase) -> list[dict]:
    rows: list[dict] = []
    print()
    for topic in HELD_OUT:
        prompt = study_prompt(topic)
        base_text = continue_text(base, tokenizer, prompt)
        tuned_text = continue_text(tuned, tokenizer, prompt)
        print(f"{topic} | base : {base_text}")
        print(f"{topic} | tuned: {tuned_text}")
        rows.append({"topic": topic, "base": base_text, "tuned": tuned_text})
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--skip-decoding", action="store_true",
                        help="Skip Step 2 and go straight to fine-tuning.")
    args = parser.parse_args()

    device = setup()
    tokenizer, base = load_base(device)
    if not args.skip_decoding:
        write_csv(compare_decoding(base, tokenizer),
                  OUT_DIR / "lab3_decoding.csv")
    tuned = fine_tune(tokenizer, device, epochs=args.epochs)
    write_csv(compare_held_out(base, tuned, tokenizer),
              OUT_DIR / "lab3_held_out.csv")


if __name__ == "__main__":
    main()
