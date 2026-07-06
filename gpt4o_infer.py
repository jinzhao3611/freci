"""Predicts framed causal claims over gold anchor events in a constrained JSON format and writes
per-document predictions consumable by eval.py (same output contract as sft_infer.py).

  export OPENAI_API_KEY=...
  python gpt4o_infer.py --split test --mode zero_shot --out runs/gpt4o/pred_zs.jsonl
  python gpt4o_infer.py --split test --mode cot       --out runs/gpt4o/pred_cot.jsonl
"""
from __future__ import annotations
import argparse, json, os
from prepare_data import to_sft            # same numbered-anchor prompt as the SFT baseline
from sft_infer import parse_claims         # tolerant JSON extractor + number->event_id mapping

COT = ("\nFirst reason step by step about who advances each causal claim, how certain it is, and "
       "who is framed as responsible; then output ONLY the final JSON list of claims.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", choices=["zero_shot", "cot"], default="zero_shot")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI()
    sp = json.load(open(f"{args.data}/splits.json"))[args.split]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fout = open(args.out, "w", encoding="utf-8")
    n = 0
    for l in open(f"{args.data}/instances.jsonl", encoding="utf-8"):
        d = json.loads(l)
        if d["topic"] not in sp:
            continue
        ex = to_sft(d)
        instruction = ex["instruction"] + (COT if args.mode == "cot" else "")
        resp = client.chat.completions.create(
            model=args.model, temperature=0,
            messages=[{"role": "user", "content": instruction + "\n\n" + ex["input"]}])
        claims = parse_claims(resp.choices[0].message.content, d["events"])
        fout.write(json.dumps({"topic": d["topic"], "doc_id": d["doc_id"], "claims": claims}) + "\n")
        n += 1
    fout.close()
    print(f"wrote {n} doc predictions -> {args.out}")

if __name__ == "__main__":
    main()
