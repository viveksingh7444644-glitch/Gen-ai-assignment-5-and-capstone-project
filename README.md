# Assignment 5: Capstone: Generative AI Creative Solution (Study Tip Generator)

**ETMMGA304 – Generative AI and Its Applications** · MCA, K.R. Mangalam University
· Capstone (lab manual pages 15–16)

A Streamlit app that gives beginners one short study tip for a topic, using GPT-2 fine-tuned with the Lab 3 workflow. The repo includes the training script, so it is self-contained.

**Settings:** GPT-2 with only the last block trained (AdamW lr 2e-5, 3 epochs); the app decodes greedily (do_sample=False, max_new_tokens 45).

## Files

```
GenAI-Assignment-5/
├── README.md
├── LAB_RECORD.md        # observation tables, results, viva answers
├── requirements.txt
├── capstone_app.py
├── lab3_gpt2_text_generation.py
├── data/                # created automatically (datasets)
└── lab_outputs/         # created automatically (results)
```

## Setup

Python 3.10 or newer is required.

```bash
git clone https://github.com/Aman13082001/GenAI-Assignment-5.git
cd GenAI-Assignment-5
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Hardware:** Runs on CPU. The first run downloads GPT-2 (about 550 MB).

### CPU or CUDA GPU

On Windows, `pip install -r requirements.txt` installs the CPU build of PyTorch.
For an NVIDIA GPU, install the CUDA build first (choose the CUDA version for
your driver at [pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Check with `python -c "import torch; print(torch.cuda.is_available())"`.

## Run

```bash
python lab3_gpt2_text_generation.py --skip-decoding   # 1. train the model
streamlit run capstone_app.py                        # 2. launch the app
```

Outputs:

- `lab_outputs/study_tip_gpt2/` (fine-tuned model used by the app)
- `lab_outputs/lab3_held_out.csv` (base vs. tuned outputs to score)

## Lab record

The observation tables, expected ranges and viva answers are in
[LAB_RECORD.md](LAB_RECORD.md). Fill the tables in from your own run.
