"""Generate framed-causal-claim JSON for a split with the SFT model, write predictions for eval.py.

  python sft_infer.py --model meta-llama/Llama-3.1-8B-Instruct --adapter runs/sft/adapter \
      --split test --out runs/sft/pred_test.jsonl
"""
from __future__ import annotations
import argparse, json, os, re

def parse_claims(text, events):
    """Pull the JSON list of claims out of the model output and map event NUMBERS back to event ids
    (using the same anchor ordering to_sft used). Tolerant of extra prose / bad rows."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m: return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for c in arr if isinstance(arr, list) else []:
        if not isinstance(c, dict): continue
        ci, ei = c.get("cause"), c.get("effect")
        if not (isinstance(ci, int) and isinstance(ei, int)): continue
        if not (0 <= ci < len(events) and 0 <= ei < len(events)): continue
        out.append({"cause_event": events[ci]["event_id"], "effect_event": events[ei]["event_id"],
                    "source": c.get("source"), "modality": c.get("modality"),
                    "targets": [{"target_text": t.get("target_text"), "framing": t.get("framing")}
                                for t in (c.get("targets") or []) if isinstance(t, dict)]})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--maxnew", type=int, default=2048)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto",
                                                 trust_remote_code=True, torch_dtype=torch.bfloat16)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    sp = json.load(open(f"{args.data}/splits.json"))[args.split]
    fout = open(args.out, "w", encoding="utf-8")
    n = 0
    for l in open(f"{args.data}/instances.jsonl", encoding="utf-8"):
        d = json.loads(l)
        if d["topic"] not in sp: continue
        from prepare_data import to_sft
        ex = to_sft(d)
        msgs = [{"role": "user", "content": ex["instruction"] + "\n\n" + ex["input"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=3072).to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.maxnew, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        claims = parse_claims(gen, d["events"])
        fout.write(json.dumps({"topic": d["topic"], "doc_id": d["doc_id"], "claims": claims}) + "\n")
        n += 1
    fout.close()
    print(f"wrote {n} doc predictions -> {args.out}")

if __name__ == "__main__":
    main()
