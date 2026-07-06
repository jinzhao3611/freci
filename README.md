# FrECI — Framing-Aware Event Causality Identification

Modeling code for **"Reframing Responsibility: Framing-Aware Event Causality Identification"** (FrECI).

FrECI extends standard Event Causality Identification: instead of only detecting whether one event
causes another, models recover **framed causal claims** that jointly encode a directed causal link and
its interpretive structure. The atomic unit of prediction is a target-specific causal assertion tuple

```
(cause, effect, source, epistemic_modality, responsibility_target, framing_effect)
```

with label sets:

| Field              | Values |
| ------------------ | ------ |
| source             | `Author`, `Target`, `Ally`, `Opponent`, `Third_Party` |
| epistemic modality | `Full_Affirmative`, `Partial_Affirmative`, `Neutral`, `Partial_Negative`, `Full_Negative` |
| framing effect     | `Blame`, `Credit`, `Undermine_Credit`, `Exonerate_Blame`, `Framing_Neutral` |

Source and modality are claim-level (shared across a causal link); responsibility targets and their
framing effects are target-specific. Evaluation is anchor-based: gold anchor event mentions are
given, and models predict all framed causal claims among them.

## Data & trained models

The dataset and trained checkpoints are on Google Drive:

> https://drive.google.com/drive/folders/1Izmvy8F_vyYDhukNWXW2s0DjzhSb9bOE

Download the topic files into `topics/`, then build the modeling splits (both `data/` and `topics/`
are git-ignored). Each `<topic>.freci.json` contains `documents` (with `sentences` and anchor
`events`) and `framed_causal_claims` (the gold tuples); see `prepare_data.py` for the exact schema.

## Setup

```bash
# install a torch build matching your CUDA first, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Pipeline

**1. Build splits** — topic-held-out (36 train / 5 dev / 10 test topics), producing
`splits.json`, `instances.jsonl`, and the per-model `sft_*.jsonl` / `pairs_*.jsonl`:

```bash
python prepare_data.py --topics topics --out data
```

**2. Train / run a model.** Three model families from Table 4, all writing predictions in the same
per-document format:

```bash
# Joint FrECI model (RoBERTa-large, multi-task heads) — the paper's best model.
# Two variants of the responsibility-target head:
#   roberta_joint.py — BIO framing-typed span tagger (lightweight)
python roberta_joint.py train --epochs 4 --out runs/joint
python roberta_joint.py infer --ckpt runs/joint/best.pt --split test --out runs/joint/pred_test.jsonl
#   roberta_srl.py   — §3.7 head: classify SRL-identified candidate agent spans (run srl_candidates.py first)
python srl_candidates.py                       # writes data/candidates.jsonl
python roberta_srl.py train --epochs 4 --out runs/joint_srl
python roberta_srl.py infer --ckpt runs/joint_srl/best.pt --split test --out runs/joint_srl/pred_test.jsonl

# SFT LLM baseline (QLoRA)
python sft_train.py  --model meta-llama/Llama-3.1-8B-Instruct --out runs/sft
python sft_infer.py  --model meta-llama/Llama-3.1-8B-Instruct --adapter runs/sft/adapter \
    --split test --out runs/sft/pred_test.jsonl

# Prompt-based GPT-4o baseline (zero-shot / CoT)
export OPENAI_API_KEY=...
python gpt4o_infer.py --split test --mode zero_shot --out runs/gpt4o/pred_zs.jsonl
python gpt4o_infer.py --split test --mode cot       --out runs/gpt4o/pred_cot.jsonl
```

**3. Evaluate** 

```bash
python eval.py --pred runs/joint/pred_test.jsonl --split test --name joint
```

## Files

| File               | Purpose |
| ------------------ | ------- |
| `prepare_data.py`   | Build topic-held-out splits and per-model training artifacts |
| `roberta_joint.py`  | Joint FrECI model; responsibility targets via a BIO framing-typed span tagger |
| `roberta_srl.py`    | Joint FrECI model; responsibility targets via classifying SRL candidate agent spans (§3.7) |
| `srl_candidates.py` | Extract candidate agent spans with the paper's PropBank SRL model (for `roberta_srl.py`) |
| `sft_train.py`      | QLoRA supervised fine-tuning of an open LLM to generate claim JSON |
| `sft_infer.py`      | Generation + prediction dump for the SFT baseline |
| `gpt4o_infer.py`    | Prompt-based GPT-4o baseline (zero-shot / CoT) |
| `eval.py`           | FrECI metrics harness |
