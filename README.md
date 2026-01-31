# Residual Stream Steering & Latent “Truth” Extraction (GPT-2 Small)

This project is a mechanistic interpretability experiment on **GPT-2 Small**.  
The goal was to see whether something like “truthfulness” shows up as a **linear direction** in the model’s residual stream — and whether that direction can be **causally manipulated** during a forward pass.

Using **TransformerLens**, I extracted a steering vector that reliably shifts the model’s behavior between:
- more factual / “news-like” reporting, and
- plausible hallucinations (confident, coherent, but wrong).

The interesting part isn’t just that steering works — it’s *what the steering direction actually contains*.

---

## What I did (high level)

1. Built paired prompts that encourage either **truthful** or **deceptive** continuations.
2. Cached internal activations with TransformerLens.
3. Computed a “truth direction” using **contrastive activation addition**:
   - take mean residual stream activations for truthful examples
   - subtract mean residual stream activations for deceptive examples
4. Injected that direction back into the residual stream using **forward hooks**
5. Ran a coefficient sweep to find the range where steering is stable vs. destructive

---

## Core concepts this project demonstrates

- **Linear Representation Hypothesis**
  - treated “truthfulness” as a direction in activation space, not a prompt trick
- **Causal intervention**
  - used residual stream hooks to modify activations mid-forward-pass
- **Contrastive Activation Addition (CAA)**
  - extracted the steering vector as a mean difference between two activation states
- **Superposition / feature entanglement**
  - the “truth” direction wasn’t pure — it interfered with other features

---

## Results: steering sweep

I swept the steering coefficient to measure where the intervention helps vs. breaks the model.

| Steering coefficient (c) | Observed behavior | Interpretation |
| --- | --- | --- |
| -2.0 | “The capital of Germany is Frankfurt.” | Plausible factual corruption (targeted lie) |
| 0.0 | Baseline continuations (often generic prose/news tone) | No intervention |
| +2.0 | More factual / news-agency style | Feature reinforcement |
| +4.0 | Semantic collapse / strange phrasing | Over-steer / saturation |

The sweep helped identify a rough “stability ceiling”: above it, the model stops behaving like a model and starts behaving like a broken decoder.

---

## Key finding: the “Geography Bias”

While extracting the truth vector, I ran into a failure mode that ended up being the most informative result of the project.

Because many of my truthful/deceptive pairs were based on **geography facts**, the extracted “truth direction” ended up entangled with a cluster of features related to:
- **Western European geography**
- **news-style reporting**
- “encyclopedic” tone

So steering the model toward “truth” didn’t just make it more accurate — it also made it *more European* in what it talked about and how it framed information.

This is a concrete example of **superposition** in small models:  
multiple concepts share representational space, so “truth” isn’t stored in a clean isolated circuit.

---

## Tools / stack

- **TransformerLens** — caching activations + residual stream hooks
- **PyTorch** — tensor ops, normalization, vector math
- **GPT-2 Small** — chosen because it’s small enough to be “surgically” interpretable

---

## Repo structure

- `steering_logic.py`
  - steering hook implementation + CAA vector extraction
- `analysis_notebook.ipynb`
  - sweep experiments, plots, and interpretation notes

---

## Why this project matters (what I learned)

Steering vectors *do* work, and you can causally shift a model’s output without changing the prompt.

But the harder (and more important) lesson is that extracted features are often **not semantically pure** — especially in smaller models. If your dataset over-represents a domain (like geography), the direction you extract may look like “truth,” but actually be a blended bundle of correlated features.

This project made that limitation visible in a measurable way.
