"""Same shared RoBERTa-large encoder and causal-link / source / modality heads as roberta_joint.py, but
responsibility targets are predicted the way §3.7 describes: by CLASSIFYING candidate agent spans
identified via SRL (rather than the BIO span tagger in roberta_joint.py). For each candidate agent
span we pool its token representations, concatenate the pair ([CLS]) representation, and predict
  (i)  select : is this candidate a responsibility target?
  (ii) framing: its framing effect, conditional on being selected.
All heads are trained jointly with a summed multi-task loss.

Candidate agent spans come from data/candidates.jsonl (run srl_candidates.py first). At training we
label the union of SRL candidates and gold target spans; at inference only SRL candidates are scored.

  train: python roberta_srl.py train --epochs 4 --out runs/joint_srl
  infer: python roberta_srl.py infer --ckpt runs/joint_srl/best.pt --split test --out runs/joint_srl/pred_test.jsonl
"""
from __future__ import annotations
import argparse, json, os
from collections import defaultdict
# reuse the shared task constants + encoder plumbing from the BIO variant (single source of truth)
from roberta_joint import SOURCE, MODALITY, FRAMING, DATA, WIN, MODEL_NAME, context_and_markers, get_tokenizer
from eval import _jacc                                    # token-Jaccard for matching candidates to gold

def load_candidates(split):
    """(topic, doc_id) -> [{sent_id, text}] SRL agent spans; empty mapping if candidates.jsonl absent."""
    cand = defaultdict(list)
    path = f"{DATA}/candidates.jsonl"
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — run srl_candidates.py first (targets will be empty).")
        return cand
    for l in open(path, encoding="utf-8"):
        d = json.loads(l)
        cand[(d["topic"], d["doc_id"])] = d.get("candidates", [])
    return cand

def locate(enc, low_text, span):
    """Token indices covering the first occurrence of `span` in the window text (char-matched)."""
    span = (span or "").strip().lower()
    if not span: return []
    s = low_text.find(span)
    if s < 0: return []
    e = s + len(span)
    return [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a != b and a < e and b > s]

def build_model(tok):
    import torch, torch.nn as nn
    from transformers import AutoModel
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(MODEL_NAME)
            self.enc.resize_token_embeddings(len(tok))
            h = self.enc.config.hidden_size
            self.link = nn.Linear(h, 2)
            self.source = nn.Linear(h, len(SOURCE))
            self.modality = nn.Linear(h, len(MODALITY))
            self.select = nn.Linear(2 * h, 2)              # candidate span -> is-responsibility-target
            self.framing = nn.Linear(2 * h, len(FRAMING))  # candidate span -> framing (conditional)
        def forward(self, input_ids, attention_mask):
            seq = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            pooled = seq[:, 0]                              # [CLS] pair representation
            return self.link(pooled), self.source(pooled), self.modality(pooled), seq
        def span_heads(self, pooled_i, seq_i, idxs):
            spanrep = seq_i.index_select(0, torch.tensor(idxs, device=seq_i.device)).mean(0)
            feat = torch.cat([pooled_i, spanrep])
            return self.select(feat), self.framing(feat)
    return M()

def collate(batch):
    import torch
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids = torch.zeros(len(batch), maxlen, dtype=torch.long)
    am = torch.zeros(len(batch), maxlen, dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"]); ids[i, :L] = torch.tensor(b["input_ids"]); am[i, :L] = 1
    return {"input_ids": ids, "attention_mask": am,
            "link": torch.tensor([b["is_causal"] for b in batch]),
            "src": torch.tensor([b["src"] for b in batch]),
            "mod": torch.tensor([b["mod"] for b in batch]),
            "cands": [b["cands_train"] for b in batch]}

def build_examples(split, tok, train=True):
    rows = [json.loads(l) for l in open(f"{DATA}/pairs_{split}.jsonl", encoding="utf-8")]
    cand_by_doc = load_candidates(split)
    exs = []
    for r in rows:
        text, enc = context_and_markers(r, tok)
        sents, cs, es = r["sentences"], r["cause"]["sent_id"], r["effect"]["sent_id"]
        lo, hi = max(0, min(cs, es) - WIN), min(len(sents), max(cs, es) + WIN + 1)
        low = text.lower()
        # SRL candidate agent spans that fall inside this pair's window (for scoring + training labels)
        srl = []
        for c in cand_by_doc.get((r["topic"], r["doc_id"]), []):
            if lo <= c["sent_id"] < hi:
                idxs = locate(enc, low, c["text"])
                if idxs: srl.append({"idxs": idxs, "text": c["text"]})
        ex = {"input_ids": enc["input_ids"], "is_causal": int(r["is_causal"]),
              "src": SOURCE.index(r["source"]) if r.get("source") in SOURCE else 0,
              "mod": MODALITY.index(r["modality"]) if r.get("modality") in MODALITY else 0,
              "cands_infer": srl, "cands_train": [],
              "meta": {"topic": r["topic"], "doc_id": r["doc_id"],
                       "cause": r["cause"]["event_id"], "effect": r["effect"]["event_id"]}}
        if train and r["is_causal"]:
            gold = r.get("targets") or []
            matched = [False] * len(gold)
            ct = []
            for c in srl:                                  # label each SRL candidate against gold targets
                gi, best = -1, 0.5
                for j, g in enumerate(gold):
                    jc = _jacc(c["text"], g.get("target_text"))
                    if jc >= best: best, gi = jc, j
                if gi >= 0: matched[gi] = True
                ct.append({"idxs": c["idxs"], "is_target": int(gi >= 0),
                           "framing": gold[gi].get("framing") if gi >= 0 else None})
            for j, g in enumerate(gold):                   # add gold spans SRL missed as positive candidates
                if matched[j]: continue
                idxs = locate(enc, low, g.get("target_text"))
                if idxs: ct.append({"idxs": idxs, "is_target": 1, "framing": g.get("framing")})
            ex["cands_train"] = ct
        exs.append(ex)
    return exs

def train(args):
    import torch, random
    from torch.utils.data import DataLoader
    tok = get_tokenizer()
    model = build_model(tok).cuda()
    tr = build_examples("train", tok, train=True)
    pos = [e for e in tr if e["is_causal"]]; neg = [e for e in tr if not e["is_causal"]]
    random.Random(0).shuffle(neg); neg = neg[: len(pos) * 2]           # 2:1 negative downsampling
    tr = pos + neg; random.Random(0).shuffle(tr)
    dl = DataLoader(tr, batch_size=args.bs, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = torch.nn.CrossEntropyLoss()
    link_ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device="cuda"))
    os.makedirs(args.out, exist_ok=True)
    for ep in range(args.epochs):
        model.train(); tot = 0
        for b in dl:
            ids, am = b["input_ids"].cuda(), b["attention_mask"].cuda()
            link, src, mod = b["link"].cuda(), b["src"].cuda(), b["mod"].cuda()
            lk, sr, md, seq = model(ids, am); pooled = seq[:, 0]
            loss = link_ce(lk, link)
            m = link == 1
            if m.any(): loss = loss + ce(sr[m], src[m]) + ce(md[m], mod[m])
            sel_l, sel_y, fr_l, fr_y = [], [], [], []      # candidate span heads (only on causal pairs)
            for i, cl in enumerate(b["cands"]):
                if link[i].item() != 1: continue
                for c in cl:
                    if not c["idxs"]: continue
                    s_log, f_log = model.span_heads(pooled[i], seq[i], c["idxs"])
                    sel_l.append(s_log); sel_y.append(c["is_target"])
                    if c["is_target"] == 1 and c["framing"] in FRAMING:
                        fr_l.append(f_log); fr_y.append(FRAMING.index(c["framing"]))
            if sel_l: loss = loss + ce(torch.stack(sel_l), torch.tensor(sel_y, device="cuda"))
            if fr_l:  loss = loss + ce(torch.stack(fr_l), torch.tensor(fr_y, device="cuda"))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        print(f"epoch {ep}: loss {tot/max(len(dl),1):.3f}")
        torch.save({"model": model.state_dict()}, f"{args.out}/best.pt")
    print("saved", f"{args.out}/best.pt")

def infer(args):
    import torch
    tok = get_tokenizer()
    model = build_model(tok).cuda()
    model.load_state_dict(torch.load(args.ckpt)["model"]); model.eval()
    exs = build_examples(args.split, tok, train=False)
    by_doc = defaultdict(list)
    with torch.no_grad():
        for e in exs:
            ids = torch.tensor([e["input_ids"]]).cuda(); am = torch.ones_like(ids)
            lk, sr, md, seq = model(ids, am); pooled = seq[:, 0]
            if torch.softmax(lk, dim=-1)[0, 1].item() < 0.35: continue
            targets = []
            for c in e["cands_infer"]:
                s_log, f_log = model.span_heads(pooled[0], seq[0], c["idxs"])
                if s_log.argmax().item() == 1:
                    targets.append({"target_text": c["text"], "framing": FRAMING[f_log.argmax().item()]})
            by_doc[(e["meta"]["topic"], e["meta"]["doc_id"])].append(
                {"cause_event": e["meta"]["cause"], "effect_event": e["meta"]["effect"],
                 "source": SOURCE[sr.argmax(-1).item()], "modality": MODALITY[md.argmax(-1).item()],
                 "targets": targets})
    fout = open(args.out, "w", encoding="utf-8")
    sp = json.load(open(f"{DATA}/splits.json"))[args.split]
    for l in open(f"{DATA}/instances.jsonl", encoding="utf-8"):
        d = json.loads(l)
        if d["topic"] in sp:
            fout.write(json.dumps({"topic": d["topic"], "doc_id": d["doc_id"],
                                   "claims": by_doc.get((d["topic"], d["doc_id"]), [])}) + "\n")
    fout.close(); print("wrote", args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--epochs", type=int, default=4); t.add_argument("--bs", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-5); t.add_argument("--out", default="runs/joint_srl")
    i = sub.add_parser("infer"); i.add_argument("--ckpt", required=True); i.add_argument("--split", default="test")
    i.add_argument("--out", required=True)
    a = ap.parse_args()
    (train if a.cmd == "train" else infer)(a)
