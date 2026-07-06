"""Convert per-topic .freci.json files into modeling-ready artifacts for both models.

Outputs (under data/):
  splits.json                  topic-held-out split (36 train / 5 dev / 10 test)
  instances.jsonl              one row per DOCUMENT: sentences, anchor events, gold claim tuples
  sft_{train,dev,test}.jsonl   {"instruction","input","output"} for QLoRA SFT (generate claim JSON)
  pairs_{train,dev,test}.jsonl candidate ordered anchor pairs (window) w/ labels for the RoBERTa joint model

The FRECI task (anchor-based): given a document + gold anchor event mentions, predict the set of
target-specific tuples (cause, effect, source, modality, target, framing).
"""
from __future__ import annotations
import json, glob, os, random, argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
# point --topics at your folder of <topic>.freci.json files (see README for the schema / data link)
DEFAULT_TOPICS = os.path.join(HERE, "topics")
WINDOW = 3            # ordered pairs whose sentences are within this many sentences
SEED = 13

def load_topics(topics_dir):
    topics = {}
    for fp in sorted(glob.glob(f"{topics_dir}/*.freci.json")):
        d = json.load(open(fp, encoding="utf-8"))
        topics[d["topic"] if d.get("topic") else os.path.basename(fp)] = d
    return topics

def make_split(topic_names):
    names = sorted(topic_names)
    random.Random(SEED).shuffle(names)
    test, dev, train = names[:10], names[10:15], names[15:]
    return {"train": sorted(train), "dev": sorted(dev), "test": sorted(test)}

def doc_instances(topic_key, d):
    """One instance per document: its sentences, anchor events, and gold claim tuples."""
    claims_by_doc = defaultdict(list)
    for c in d.get("framed_causal_claims", []):
        claims_by_doc[c["doc_id"]].append(c)
    out = []
    for doc in d.get("documents", []):
        did = doc["doc_id"]
        events = [{"event_id": e["event_id"], "trigger": e.get("trigger", ""),
                   "sent_id": e.get("sent_id", -1)} for e in doc.get("events", [])]
        gold = []
        for c in claims_by_doc.get(did, []):
            gold.append({
                "cause_event": c["cause_event"], "effect_event": c["effect_event"],
                "source": c.get("source"), "modality": c.get("modality"),
                "targets": [{"target_text": t.get("target_text"), "framing": t.get("framing")}
                            for t in (c.get("targets") or [])],
                "evidence_quote": c.get("evidence_quote", ""),
            })
        out.append({"topic": topic_key, "doc_id": did, "language": doc.get("language"),
                    "sentences": doc.get("sentences", []), "events": events, "gold_claims": gold})
    return out

def to_sft(inst):
    """Instruction-tuning pair. Events are referred to by NUMBER (index into the anchor list), not by
    their hash id — LLMs can echo a small integer reliably but not `EVENT_9234f826`. sft_infer maps
    the numbers back to event ids using the same ordering."""
    events = inst["events"]
    id2idx = {e["event_id"]: i for i, e in enumerate(events)}
    sents = "\n".join(f"[{i}] {s}" for i, s in enumerate(inst["sentences"]))
    anchors = "\n".join(f"[{i}] \"{e['trigger']}\"  (sentence {e['sent_id']})"
                        for i, e in enumerate(events))
    instruction = (
        "You are given a document (numbered sentences) and a numbered list of ANCHOR events. "
        "Identify every directed causal relation between anchor events and output the framed causal "
        "claims. Refer to events by their NUMBER. For each causal relation give: cause (event number), "
        "effect (event number), source (Author/Target/Ally/Opponent/Third_Party), modality "
        "(Full_Affirmative/Partial_Affirmative/Neutral/Partial_Negative/Full_Negative), and targets "
        "(a list of {target_text, framing}, framing in Blame/Credit/Undermine_Credit/Exonerate_Blame/"
        "Framing_Neutral; empty list if no actor is named). Output ONLY a JSON list of claims, e.g. "
        '[{"cause": 0, "effect": 3, "source": "Author", "modality": "Full_Affirmative", "targets": []}]')
    inp = f"DOCUMENT:\n{sents}\n\nANCHOR EVENTS (refer to these by number):\n{anchors}"
    claims = []
    for c in inst["gold_claims"]:
        if c["cause_event"] in id2idx and c["effect_event"] in id2idx:
            claims.append({"cause": id2idx[c["cause_event"]], "effect": id2idx[c["effect_event"]],
                           "source": c["source"], "modality": c["modality"], "targets": c["targets"]})
    return {"instruction": instruction, "input": inp, "output": json.dumps(claims, ensure_ascii=False)}

def to_pairs(inst):
    """Candidate ordered anchor pairs within WINDOW sentences, labelled from gold."""
    gold = {}
    for c in inst["gold_claims"]:
        gold[(c["cause_event"], c["effect_event"])] = c  # directed
    evs = inst["events"]
    rows = []
    for a in evs:
        for b in evs:
            if a["event_id"] == b["event_id"]:
                continue
            if a["sent_id"] < 0 or b["sent_id"] < 0 or abs(a["sent_id"] - b["sent_id"]) > WINDOW:
                continue
            c = gold.get((a["event_id"], b["event_id"]))
            rows.append({
                "topic": inst["topic"], "doc_id": inst["doc_id"],
                "cause": a, "effect": b,
                "sentences": inst["sentences"],
                "is_causal": c is not None,
                "source": c["source"] if c else None,
                "modality": c["modality"] if c else None,
                "targets": c["targets"] if c else [],
            })
    return rows

def main():
    ap = argparse.ArgumentParser(description="Build FRECI train/dev/test from a folder of <topic>.freci.json")
    ap.add_argument("--topics", default=DEFAULT_TOPICS, help="folder of <topic>.freci.json files")
    ap.add_argument("--out", default=os.path.join(HERE, "data"), help="output folder for the splits")
    args = ap.parse_args()
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    topics = load_topics(args.topics)
    split = make_split(list(topics.keys()))
    json.dump(split, open(f"{OUT}/splits.json", "w"), indent=2)
    topic2split = {t: s for s, ts in split.items() for t in ts}

    inst_f = open(f"{OUT}/instances.jsonl", "w", encoding="utf-8")
    sft = {s: open(f"{OUT}/sft_{s}.jsonl", "w", encoding="utf-8") for s in ("train", "dev", "test")}
    prf = {s: open(f"{OUT}/pairs_{s}.jsonl", "w", encoding="utf-8") for s in ("train", "dev", "test")}
    n_docs = n_claims = n_pairs = n_pos = 0
    for tkey, d in topics.items():
        s = topic2split[tkey]
        for inst in doc_instances(tkey, d):
            inst_f.write(json.dumps(inst, ensure_ascii=False) + "\n")
            n_docs += 1; n_claims += len(inst["gold_claims"])
            sft[s].write(json.dumps(to_sft(inst), ensure_ascii=False) + "\n")
            for r in to_pairs(inst):
                prf[s].write(json.dumps(r, ensure_ascii=False) + "\n")
                n_pairs += 1; n_pos += int(r["is_causal"])
    inst_f.close()
    for f in list(sft.values()) + list(prf.values()): f.close()
    print(f"topics={len(topics)}  docs={n_docs}  gold_claims={n_claims}")
    print(f"split: train={len(split['train'])} dev={len(split['dev'])} test={len(split['test'])} topics")
    print(f"candidate pairs={n_pairs}  causal={n_pos} ({100*n_pos/max(n_pairs,1):.1f}%)")
    for s in ("train", "dev", "test"):
        nc = sum(len(json.loads(l)["gold_claims"]) for l in open(f"{OUT}/instances.jsonl") if json.loads(l)["topic"] in split[s])
        nd = sum(1 for l in open(f"{OUT}/instances.jsonl") if json.loads(l)["topic"] in split[s])
        print(f"  {s}: {nd} docs, {nc} gold claims")

if __name__ == "__main__":
    main()
