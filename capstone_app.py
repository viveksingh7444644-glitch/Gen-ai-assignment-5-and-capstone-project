"""Capstone - Study Tip Generator (Streamlit demonstration).

ETMMGA304 Laboratory Manual, pages 15-16.

Loads the GPT-2 model fine-tuned in Lab 3 from lab_outputs/study_tip_gpt2
and generates a deterministic study tip for a user-supplied topic.

Run (after lab3_gpt2_text_generation.py):
    streamlit run capstone_app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedModel, PreTrainedTokenizerBase)

MODEL_DIR = Path(__file__).resolve().parent / "lab_outputs" / "study_tip_gpt2"
MAX_TOPIC_CHARS = 120
MAX_NEW_TOKENS = 45


@st.cache_resource
def load_model(folder: str) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load tokenizer and model once per Streamlit server process."""
    tok = AutoTokenizer.from_pretrained(folder)
    net = AutoModelForCausalLM.from_pretrained(folder).eval()
    return tok, net


def generate_tip(tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel,
                 topic: str) -> str:
    """Greedy-decode a study tip for ``topic`` (same format as training)."""
    inputs = tokenizer(f"Topic: {topic}\nStudy tip:", return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False,
                                max_new_tokens=MAX_NEW_TOKENS,
                                pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, inputs.input_ids.shape[1]:],
                            skip_special_tokens=True).strip()


def main() -> None:
    st.set_page_config(page_title="Study Tip Generator")
    st.title("Study Tip Generator")
    st.caption("AI-generated study tips for instructor review. Outputs may be "
               "inaccurate; a teacher should check them before students use "
               "them.")

    if not (MODEL_DIR / "config.json").exists():
        st.error("Fine-tuned model not found. Run "
                 "`python lab3_gpt2_text_generation.py` first so that "
                 f"`{MODEL_DIR}` exists.")
        st.stop()

    topic = st.text_input("Enter a study topic", max_chars=MAX_TOPIC_CHARS)
    if st.button("Generate"):
        if not topic.strip():
            st.warning("Enter a study topic first.")
            st.stop()
        tokenizer, model = load_model(str(MODEL_DIR))
        with st.spinner("Generating..."):
            tip = generate_tip(tokenizer, model, topic.strip())
        st.write(tip or "No text was generated. Try a more specific topic.")


main()
