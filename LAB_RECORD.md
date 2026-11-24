# Lab Record: Assignment 5, ETMMGA304 Generative AI and Its Applications

**Student:** ____________________  **Roll no.:** ____________  **Programme:** MCA

Fill in each blank table from **your own run**. The *expected ranges* below
are approximate values for the default settings, to help you spot a broken
run. They are not results to copy. Values change with hardware, library
versions and random seeds.

Write down the following for every experiment:

| Field | Value |
|-------|-------|
| Date | |
| Device (CPU / GPU model) | |
| Python / PyTorch / Transformers versions | |
| Seed | 42 |

---

## Capstone: Generative AI Creative Solution (Study Tip Generator)

**Define.** Beginners often do not know *how* to start studying a new
topic. The app takes a topic and returns one short, actionable study tip.
**Success criteria:** mean adapted score ≥ 4/6 on at least 10 held-out
topics, and higher than the base model.

**Prepare.** Instructor-approved (topic, tip) pairs, with separate training
and held-out topic lists that do not overlap. Record the source,
permissions and split sizes.

**Adapt.** GPT-2 with only the last block trained (Lab 3 workflow).
Settings: lr 2e-5, 3 or more epochs, AdamW. Keep the loss log and the
`lab_outputs/study_tip_gpt2` checkpoint.

**Demonstrate.** `streamlit run capstone_app.py`

### Evaluation record (≥ 10 held-out topics; each criterion 0–2)

| Topic | Model | Relevance | Readability | Usefulness | Total / 6 | Notes |
|-------|-------|-----------|-------------|------------|-----------|-------|
| recursion | Base | | | | | |
| recursion | Adapted | | | | | |
| matrices | Base | | | | | |
| matrices | Adapted | | | | | |
| binary search | Base | | | | | |
| binary search | Adapted | | | | | |
| linked lists | Base | | | | | |
| linked lists | Adapted | | | | | |
| derivatives | Base | | | | | |
| derivatives | Adapted | | | | | |
| SQL joins | Base | | | | | |
| SQL joins | Adapted | | | | | |
| hash tables | Base | | | | | |
| hash tables | Adapted | | | | | |
| logarithms | Base | | | | | |
| logarithms | Adapted | | | | | |
| stacks and queues | Base | | | | | |
| stacks and queues | Adapted | | | | | |
| Big-O notation | Base | | | | | |
| Big-O notation | Adapted | | | | | |
| **Mean** | **Base** | | | | | |
| **Mean** | **Adapted** | | | | | |

**Failure analysis (at least one).** Example topic: ______. Output:
"______". Why it fails: ______ (for example, a generic tip that would fit
any topic, a factually wrong statement, or text that continues with a new
"Topic:" line).

**Conditions for use.** For teacher-reviewed drafting only. Not for
unsupervised student use. English topics only. Short factual topics only.

### Demonstration checks

| Check | Pass? | Notes |
|-------|-------|-------|
| Empty topic shows the "Enter a study topic first." warning | | |
| Missing model folder shows an error and stops | | |
| Held-out topic 1 gives an understandable, useful tip | | |
| Held-out topic 2 gives an understandable, useful tip | | |
| One weak output presented, with how a reviewer would handle it | | |

### Privacy, bias and limitations (disclosure)

- **Disclosure:** the app labels every tip as AI-generated and meant for
  instructor review.
- **Privacy:** the model runs locally. Topics are not logged or sent to a
  third party. Do not enter personal data.
- **Bias:** GPT-2 was trained on WebText (Reddit-linked pages), which
  leans towards English, Western and male-authored content. Tips may assume
  particular resources or learning styles.
- **Limitations:** a 124M-parameter model can state false facts with
  confidence. Output length is fixed at 45 tokens. Greedy decoding always
  gives the same answer for a topic.

### Viva answers

1. **Why did you select this dataset and model?**
   GPT-2 is open, small enough to fine-tune on CPU or a free Colab GPU, and
   already writes fluent English, so only task formatting needs to be
   learned. Training just the last block cuts compute and the risk of
   over-fitting a small dataset. Instructor-approved (topic, tip) pairs give
   the model data that is accurate, appropriate and properly licensed.

2. **Which results show whether adaptation improved the outputs?**
   The paired comparison on the same held-out topics: the mean total score
   out of 6 for base and adapted models, per-criterion differences, and raw
   example outputs. Training loss alone is not evidence, because it is
   measured on seen examples. A higher held-out mean, with consistent gains
   across most topics, shows real improvement.

3. **What human review would you require before students use the app?**
   An instructor checks each generated tip for accuracy, clarity and
   suitability before it reaches students, using an approved list or a
   moderation queue. Spot-checks are repeated whenever the model or data
   changes, and students get a way to report bad tips. The interface states
   clearly that the content is AI-generated.
