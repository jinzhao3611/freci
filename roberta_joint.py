"""Heads (all off the same encoding of a candidate cause->effect pair in its sentence-window context):
  - causal link      : binary, is this directed anchor pair causal?
  - source           : 5-way claim-level (Author/Target/Ally/Opponent/Third_Party)
  - modality         : 5-way claim-level
  - target + framing : token-level BIO tagger with framing-typed spans (O + B/I x 5 framings),
                       a lightweight stand-in for the paper's SRL-candidate target-span + framing heads
Trained jointly with a summed multi-task loss on causal pairs (attribute losses masked to causal pairs).

  train: python roberta_joint.py train --epochs 4 --out runs/joint
  infer: python roberta_joint.py infer --ckpt runs/joint/best.pt --split test --out runs/joint/pred_test.jsonl
"""
from __future__ import annotations
import argparse, json, os
SOURCE = ["Author", "Target", "Ally", "Opponent", "Third_Party"]
MODALITY = ["Full_Affirmative", "Partial_Affirmative", "Neutral", "Partial_Negative", "Full_Negative"]
FRAMING = ["Blame", "Credit", "Undermine_Credit", "Exonerate_Blame", "Framing_Neutral"]
BIO = ["O"] + [f"{p}-{f}" for f in FRAMING for p in ("B", "I")]        # 11 tags
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WIN = 2                                                                # context sentences each side of the pair
MODEL_NAME = "FacebookAI/roberta-large"

def context_and_markers(row, tok, maxlen=256):
    """Build the input: window of sentences around cause+effect, with [C]/[E] markers on triggers."""
    sents, cs, es = row["sentences"], row["cause"]["sent_id"], row["effect"]["sent_id"]
    lo, hi = max(0, min(cs, es) - WIN), min(len(sents), max(cs, es) + WIN + 1)
    text = " ".join(f"[C] {sents[i]} [/C]" if i == cs else (f"[E] {sents[i]} [/E]" if i == es else sents[i])
                    for i in range(lo, hi))
    enc = tok(text, truncation=True, max_length=maxlen, return_offsets_mapping=True)
    return text, enc

def bio_labels(text, enc, targets):
    """Char-match each gold target_text in the window text -> BIO-framing token labels."""
    n = len(enc["input_ids"]); labels = [0] * n
    low = text.lower()
    for t in targets or []:
        tt = (t.get("target_text") or "").strip().lower()
        fr = t.get("framing")
        if not tt or fr not in FRAMING: continue
        s = low.find(tt)
        if s < 0: continue
        e = s + len(tt)
        first = True
        for i, (a, b) in enumerate(enc["offset_mapping"]):
            if a == b: continue
            if a < e and b > s:
                labels[i] = BIO.index(("B-" if first else "I-") + fr); first = False
    return labels

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
            self.tagger = nn.Linear(h, len(BIO))
        def forward(self, input_ids, attention_mask):
            out = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            pooled = out[:, 0]                                   # [CLS]
            return (self.link(pooled), self.source(pooled), self.modality(pooled), self.tagger(out))
    return M()

def collate(batch, tok):
    import torch
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids = torch.zeros(len(batch), maxlen, dtype=torch.long)
    am = torch.zeros(len(batch), maxlen, dtype=torch.long)
    tags = torch.full((len(batch), maxlen), -100, dtype=torch.long)
    link = torch.tensor([b["is_causal"] for b in batch], dtype=torch.long)
    src = torch.tensor([b["src"] for b in batch], dtype=torch.long)
    mod = torch.tensor([b["mod"] for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"]); am[i, :L] = 1
        if b["is_causal"]:
            tags[i, :L] = torch.tensor(b["tags"])
    return {"input_ids": ids, "attention_mask": am, "tags": tags, "link": link, "src": src, "mod": mod}

def build_examples(split, tok):
    rows = [json.loads(l) for l in open(f"{DATA}/pairs_{split}.jsonl", encoding="utf-8")]
    exs = []
    for r in rows:
        text, enc = context_and_markers(r, tok)
        ex = {"input_ids": enc["input_ids"], "is_causal": int(r["is_causal"]),
              "src": SOURCE.index(r["source"]) if r.get("source") in SOURCE else 0,
              "mod": MODALITY.index(r["modality"]) if r.get("modality") in MODALITY else 0,
              "tags": bio_labels(text, enc, r.get("targets")) if r["is_causal"] else [],
              "meta": {"topic": r["topic"], "doc_id": r["doc_id"],
                       "cause": r["cause"]["event_id"], "effect": r["effect"]["event_id"]}}
        exs.append(ex)
    return exs

def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    tok.add_special_tokens({"additional_special_tokens": ["[C]", "[/C]", "[E]", "[/E]"]})
    return tok

def train(args):
    import torch, random
    from torch.utils.data import DataLoader
    tok = get_tokenizer()
    model = build_model(tok).cuda()
    tr = build_examples("train", tok)
    # class balance: downsample negatives (non-causal pairs dominate ~96%)
    pos = [e for e in tr if e["is_causal"]]; neg = [e for e in tr if not e["is_causal"]]
    random.Random(0).shuffle(neg); neg = neg[: len(pos) * 2]          # 2:1 (was 4:1 -> collapsed)
    tr = pos + neg; random.Random(0).shuffle(tr)
    dl = DataLoader(tr, batch_size=args.bs, shuffle=True, collate_fn=lambda b: collate(b, tok))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = torch.nn.CrossEntropyLoss()
    link_ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device="cuda"))  # upweight causal
    tag_ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    os.makedirs(args.out, exist_ok=True)
    for ep in range(args.epochs):
        model.train(); tot = 0
        for step, b in enumerate(dl):
            b = {k: v.cuda() for k, v in b.items()}
            lk, sr, md, tg = model(b["input_ids"], b["attention_mask"])
            loss = link_ce(lk, b["link"])
            m = b["link"] == 1                                        # attribute losses only on causal pairs
            if m.any():
                loss = loss + ce(sr[m], b["src"][m]) + ce(md[m], b["mod"][m])
                loss = loss + tag_ce(tg[m].reshape(-1, len(BIO)), b["tags"][m].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        print(f"epoch {ep}: loss {tot/max(len(dl),1):.3f}")
        torch.save({"model": model.state_dict()}, f"{args.out}/best.pt")
    print("saved", f"{args.out}/best.pt")

def decode_bio(ids, tag_ids, tok):
    """Turn a BIO-framing tag sequence back into (target_text, framing) spans."""
    spans, cur, cur_f = [], [], None
    toks = tok.convert_ids_to_tokens(ids)
    for t, tg in zip(toks, tag_ids):
        name = BIO[tg]
        if name.startswith("B-"):
            if cur: spans.append((tok.convert_tokens_to_string(cur).strip(), cur_f))
            cur, cur_f = [t], name[2:]
        elif name.startswith("I-") and cur_f == name[2:]:
            cur.append(t)
        else:
            if cur: spans.append((tok.convert_tokens_to_string(cur).strip(), cur_f)); cur, cur_f = [], None
    if cur: spans.append((tok.convert_tokens_to_string(cur).strip(), cur_f))
    return [{"target_text": s, "framing": f} for s, f in spans if s]

def infer(args):
    import torch
    from collections import defaultdict
    tok = get_tokenizer()
    model = build_model(tok).cuda()
    model.load_state_dict(torch.load(args.ckpt)["model"]); model.eval()
    exs = build_examples(args.split, tok)
    by_doc = defaultdict(list)
    with torch.no_grad():
        for e in exs:
            ids = torch.tensor([e["input_ids"]]).cuda(); am = torch.ones_like(ids)
            lk, sr, md, tg = model(ids, am)
            if torch.softmax(lk, dim=-1)[0, 1].item() < 0.35:        # recall-friendly threshold (not argmax)
                continue
            tags = tg[0].argmax(-1).tolist()
            claim = {"cause_event": e["meta"]["cause"], "effect_event": e["meta"]["effect"],
                     "source": SOURCE[sr.argmax(-1).item()], "modality": MODALITY[md.argmax(-1).item()],
                     "targets": decode_bio(e["input_ids"], tags, tok)}
            by_doc[(e["meta"]["topic"], e["meta"]["doc_id"])].append(claim)
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
    t.add_argument("--lr", type=float, default=1e-5); t.add_argument("--out", default="runs/joint")
    i = sub.add_parser("infer"); i.add_argument("--ckpt", required=True); i.add_argument("--split", default="test")
    i.add_argument("--out", required=True)
    a = ap.parse_args()
    (train if a.cmd == "train" else infer)(a)
