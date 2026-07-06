"""QLoRA SFT of an open LLM to generate framed-causal-claim JSON.

  python sft_train.py --model meta-llama/Llama-3.1-8B-Instruct --epochs 3 --out runs/sft_llama

Trains on data/sft_train.jsonl ({instruction,input,output}); dev for eval. Uses 4-bit QLoRA so an
8B fits on one 48GB GPU. Inference/scoring is in sft_infer.py.
"""
from __future__ import annotations
import argparse, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--maxlen", type=int, default=2048)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb,
                                                 device_map="auto", trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def to_text(ex):
        msgs = [{"role": "user", "content": ex["instruction"] + "\n\n" + ex["input"]},
                {"role": "assistant", "content": ex["output"]}]
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}
    ds = load_dataset("json", data_files={"train": f"{args.data}/sft_train.jsonl",
                                          "dev": f"{args.data}/sft_dev.jsonl"})
    ds = ds.map(to_text, remove_columns=ds["train"].column_names)

    cfg = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=args.bs,
                    gradient_accumulation_steps=args.accum, learning_rate=args.lr, bf16=True,
                    logging_steps=10, save_strategy="epoch", eval_strategy="epoch",
                    max_seq_length=args.maxlen, packing=False, dataset_text_field="text", report_to="none",
                    warmup_ratio=0.03, lr_scheduler_type="cosine", gradient_checkpointing=True)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["dev"],
                         processing_class=tok)
    trainer.train()
    trainer.save_model(args.out + "/adapter")
    tok.save_pretrained(args.out + "/adapter")
    print("saved adapter to", args.out + "/adapter")

if __name__ == "__main__":
    main()
