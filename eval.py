"""Predictions + gold are per-document lists of claim tuples:
  {"cause_event","effect_event","source","modality","targets":[{"target_text","framing"}]}

Metrics:
  Causal Link P/R/F1     over directed (cause,effect) anchor pairs
  Source / Modality      claim-level macro-F1, on causal links that matched gold
  Responsibility Target  span-F1 via token-Jaccard >= 0.5 one-to-one match (we have text, not offsets)
  Framing Effect         macro-F1 over matched targets
  Full-Claim EM          gold link recovered iff cause,effect,source,modality AND full target+framing set match
"""
from __future__ import annotations
import json, re, argparse, os

SOURCE = ["Author", "Target", "Ally", "Opponent", "Third_Party"]
MODALITY = ["Full_Affirmative", "Partial_Affirmative", "Neutral", "Partial_Negative", "Full_Negative"]
FRAMING = ["Blame", "Credit", "Undermine_Credit", "Exonerate_Blame", "Framing_Neutral"]

def _toks(s): return set(re.sub(r"[^\w\s]", " ", (s or "").lower()).split())
def _jacc(a, b):
    A, B = _toks(a), _toks(b)
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def _macro_f1(pairs, labels):
    """pairs: list of (gold_label, pred_label). Macro-F1 over labels present in gold."""
    f1s = []
    for L in labels:
        if not any(g == L for g, _ in pairs): continue
        tp = sum(1 for g, p in pairs if g == L and p == L)
        fp = sum(1 for g, p in pairs if g != L and p == L)
        fn = sum(1 for g, p in pairs if g == L and p != L)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0

def _match_targets(gts, pts, thr=0.5):
    used, matched, ug = set(), [], []
    for g in gts:
        best, bestj = -1, thr
        for pi, p in enumerate(pts):
            if pi in used: continue
            j = _jacc(g.get("target_text"), p.get("target_text"))
            if j >= bestj: bestj, best = j, pi
        if best >= 0: used.add(best); matched.append((g, pts[best]))
        else: ug.append(g)
    up = [p for pi, p in enumerate(pts) if pi not in used]
    return matched, ug, up

def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

def score(docs):
    """docs: list of (gold_claims, pred_claims)."""
    Ltp = Lfp = Lfn = Ttp = Tfp = Tfn = emc = emt = 0
    src, mod, frm = [], [], []
    for gold, pred in docs:
        gm = {(c["cause_event"], c["effect_event"]): c for c in gold}
        pm = {(c["cause_event"], c["effect_event"]): c for c in pred}
        gk, pk = set(gm), set(pm)
        Ltp += len(gk & pk); Lfp += len(pk - gk); Lfn += len(gk - pk); emt += len(gk)
        for k in gk & pk:
            g, p = gm[k], pm[k]
            src.append((g.get("source"), p.get("source")))
            mod.append((g.get("modality"), p.get("modality")))
            matched, ug, up = _match_targets(g.get("targets") or [], p.get("targets") or [])
            Ttp += len(matched); Tfn += len(ug); Tfp += len(up)
            for gt, pt in matched: frm.append((gt.get("framing"), pt.get("framing")))
            full = (g.get("source") == p.get("source") and g.get("modality") == p.get("modality")
                    and not ug and not up and all(a.get("framing") == b.get("framing") for a, b in matched))
            emc += int(full)
    lp, lr, lf = _prf(Ltp, Lfp, Lfn)
    _, _, tf = _prf(Ttp, Tfp, Tfn)
    return {
        "causal_link_P": round(100 * lp, 1), "causal_link_R": round(100 * lr, 1), "causal_link_F1": round(100 * lf, 1),
        "source_macroF1": round(100 * _macro_f1(src, SOURCE), 1),
        "modality_macroF1": round(100 * _macro_f1(mod, MODALITY), 1),
        "target_F1": round(100 * tf, 1),
        "framing_macroF1": round(100 * _macro_f1(frm, FRAMING), 1),
        "full_claim_EM": round(100 * emc / max(emt, 1), 1),
        "n_gold_links": emt,
    }

def load_gold(split="test", data=None):
    data = data or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    sp = json.load(open(f"{data}/splits.json"))[split]
    gold = {}
    for l in open(f"{data}/instances.jsonl", encoding="utf-8"):
        d = json.loads(l)
        if d["topic"] in sp:
            gold[(d["topic"], d["doc_id"])] = d["gold_claims"]
    return gold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="jsonl: {topic,doc_id,claims:[...]}")
    ap.add_argument("--split", default="test")
    ap.add_argument("--name", default="model")
    args = ap.parse_args()
    gold = load_gold(args.split)
    preds = {}
    for l in open(args.pred, encoding="utf-8"):
        d = json.loads(l); preds[(d["topic"], d["doc_id"])] = d.get("claims", [])
    docs = [(g, preds.get(k, [])) for k, g in gold.items()]
    res = score(docs)
    print(f"=== {args.name}  ({args.split}, {len(docs)} docs, {res['n_gold_links']} gold links) ===")
    for k, v in res.items(): print(f"  {k:18s} {v}")

if __name__ == "__main__":
    main()
