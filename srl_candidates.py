"""For every anchor event we run the PropBank SRL seq2seq model used in the paper
(cu-kairos/propbank_srl_seq2seq_t5_large), marking the event trigger as the predicate, and keep the
ARG-0 (agent) argument as a candidate responsibility target. roberta_srl.py then classifies these
candidates as targets and assigns framing effects.

Model I/O (per the model card):
  input : "SRL for [put]: That fund was [put] together by Blackstone Group ."
  output: "ARG-1: That fund | ARG-2: together | ARG-0: by Blackstone Group"

  python srl_candidates.py            # reads data/instances.jsonl -> data/candidates.jsonl
"""
from __future__ import annotations
import argparse, json, os

MODEL_NAME = "cu-kairos/propbank_srl_seq2seq_t5_large"
AGENT_ROLES = ("ARG-0", "ARG-1")   # ARG-0 = agent; ARG-1 kept as a fallback agent-like argument

def mark_predicate(sentence, trigger):
    """Wrap the first occurrence of the trigger in [ ] as the SRL predicate marker."""
    i = sentence.find(trigger) if trigger else -1
    return sentence if i < 0 else sentence[:i] + "[" + trigger + "]" + sentence[i + len(trigger):]

def parse_roles(out):
    """'ARG-1: That fund | ARG-0: by Blackstone Group' -> {'ARG-1': 'That fund', 'ARG-0': 'by ...'}"""
    roles = {}
    for chunk in out.split("|"):
        role, sep, span = chunk.partition(":")
        if sep and role.strip() and span.strip():
            roles[role.strip()] = span.strip()
    return roles

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--maxnew", type=int, default=128)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval()
    if torch.cuda.is_available(): model = model.cuda()

    fout = open(f"{args.data}/candidates.jsonl", "w", encoding="utf-8")
    n_docs = n_cand = 0
    for l in open(f"{args.data}/instances.jsonl", encoding="utf-8"):
        d = json.loads(l)
        sents = d["sentences"]
        seen, cands = set(), []
        for e in d["events"]:
            sid, trig = e.get("sent_id", -1), e.get("trigger", "")
            if not (0 <= sid < len(sents)): continue
            inp = f"SRL for [{trig}]: " + mark_predicate(sents[sid], trig)
            ids = tok(inp, return_tensors="pt", truncation=True, max_length=256).to(model.device)
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=args.maxnew)
            roles = parse_roles(tok.decode(gen[0], skip_special_tokens=True))
            for r in AGENT_ROLES:
                span = roles.get(r)
                key = (sid, (span or "").lower())
                if span and key not in seen:
                    seen.add(key); cands.append({"sent_id": sid, "text": span})
        fout.write(json.dumps({"topic": d["topic"], "doc_id": d["doc_id"], "candidates": cands},
                              ensure_ascii=False) + "\n")
        n_docs += 1; n_cand += len(cands)
    fout.close()
    print(f"wrote candidates for {n_docs} docs ({n_cand} agent spans) -> {args.data}/candidates.jsonl")

if __name__ == "__main__":
    main()
