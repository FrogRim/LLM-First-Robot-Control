from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--adapter_dir", default="./droid-physics-qwen14b-qlora")
    parser.add_argument("--out_dir", default="./droid-physics-qwen14b-merged")
    args = parser.parse_args()

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.out_dir)

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.save_pretrained(args.out_dir)
    print(f"✅ Merged model saved to: {args.out_dir}")


if __name__ == "__main__":
    main()


