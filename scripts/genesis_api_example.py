import argparse
from src.genesis_api import GenesisClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Genesis API 호출 예제")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    client = GenesisClient()
    resp = client.generate_text(
        model_id=args.model_id,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=0.7,
        top_p=0.95,
    )
    print(resp)


if __name__ == "__main__":
    main()
