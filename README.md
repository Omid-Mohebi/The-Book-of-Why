# The Book of Why
### A Python implementation of Pearl's causal hierarchy

---

## Overview

This project implements the three core ideas from **The Book of Why**
(Judea Pearl & Dana Mackenzie, 2018) in pure Python:

| Concept | File | What it does |
|---|---|---|
| Causal DAG | `causal_model.py` | Directed acyclic graph encoding causal assumptions |
| do-operator | `causal_model.py` → `do()` | Graph surgery: removes incoming edges to a treatment node |
| Backdoor criterion | `causal_model.py` → `find_backdoor_adjustment_sets()` | Finds valid covariate sets for causal identification |
| Backdoor adjustment | `estimator.py` | Estimates the Average Causal Effect via OLS + adjustment |
| d-separation | `causal_model.py` → `is_d_separated()` | Tests conditional independence in the DAG |

---

## Causal Story

> *"Does getting more education actually raise your income, or does
> coming from a wealthy family just make both more likely?"*

```
SES (Socioeconomic Status)
  ↓                  ↓
Education   →    Income
  ↓
JobLevel
```

**SES is a confounder.** It opens a backdoor path:

```
Education ← SES → Income
```

Without adjustment, a naive regression incorrectly attributes the SES bonus
to Education.

### Structural Equations (true data-generating process)
```
SES       ~ Normal(50, 10)
Education = 12 + 0.15·SES + noise      ← SES causes Education
JobLevel  = round(0.3·Education − 2) + noise
Income    = 20 000 + 2 000·Education   ← TRUE effect = $2,000/yr
           + 800·SES + noise           ← SES also directly causes Income
```

---

## Results

| Method | Estimated effect | True effect |
|---|---|---|
| Naive regression | ~$4,055 / year | $2,000 / year |
| Backdoor adjustment (adj. for SES) | ~$2,127 / year | $2,000 / year |

The **confounding bias ≈ $1,928** — the naive estimate nearly doubles the
real effect by absorbing the SES → Income pathway.

---

## Files

```
The Backdoor Criterion/
├── causal_model.py   — CausalDAG class (DAG, do(), backdoor, d-sep)
├── dataset.py        — Synthetic data generator
├── estimator.py      — BackdoorEstimator + naive_regression
├── visualizer.py     — DAG plot, results chart, do-operator illustration
├── main.py           — End-to-end demo (run this)
└── README.md         — This file
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Enter the project folder
cd "The Backdoor Criterion" or
cd The\ Backdoor\ Criterion

# Run the full demo
python main.py
```

Three figures are produced:
- **dag.png** — The causal DAG with colour-coded nodes
- **results.png** — Naive vs adjusted estimate vs true effect
- **do_operator.png** — Distribution shift before/after do(Education)
