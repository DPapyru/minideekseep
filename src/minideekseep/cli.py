import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .config import ModelConfig
from .data import TokenStream, collect, encode_corpus, fingerprint, train_tokenizer
from .model import MiniDeepSeek
from .training import benchmark, compare_prefill, evaluate, load_checkpoint, train


def config_for(args):
    if args.config:
        return ModelConfig.load(args.config)
    return ModelConfig.tiny() if args.tiny else ModelConfig()


def write_result(value, path=None):
    text = json.dumps(value, indent=2, ensure_ascii=False)
    print(text, flush=True)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text + "\n")


def main():
    parser = argparse.ArgumentParser(prog="minideekseep")
    parser.add_argument("--threads", type=int, default=4)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("params", "benchmark", "train"):
        p = commands.add_parser(name)
        p.add_argument("--config")
        p.add_argument("--tiny", action="store_true")
        if name == "params":
            p.add_argument("--output")
            p.add_argument("--save-config")
        else:
            p.add_argument("--device", default="cuda")
            p.add_argument("--length", type=int, default=128)
            p.add_argument("--tokens-per-step", type=int, default=4096)
            if name == "benchmark":
                p.add_argument("--steps", type=int, default=100)
                p.add_argument("--seconds", type=float, default=600)
                p.add_argument("--output", default="runs/benchmark.json")
            else:
                p.add_argument("--data", default="data/finewiki")
                p.add_argument("--run", default="runs/pretrain")
                p.add_argument("--tokens", type=int, default=1_000_000)
                p.add_argument("--max-steps", type=int)
                p.add_argument("--resume")
                p.add_argument("--lr", type=float, default=3e-4)
                p.add_argument(
                    "--initialize", help="Start a new training stage from this checkpoint"
                )
                p.add_argument("--replay", action="store_true")
                p.add_argument("--fake-quant", action="store_true")
    p = commands.add_parser("prepare")
    p.add_argument("--data", default="data/finewiki")
    p.add_argument("--source", help="Local JSONL records containing text, optional language/id/url")
    p.add_argument("--documents", type=int, default=30000)
    p.add_argument("--revision")
    p.add_argument(
        "--preview", action="store_true", help="Small current sample via dataset-viewer API"
    )
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--encode-only", action="store_true")
    p = commands.add_parser("evaluate")
    p.add_argument("checkpoint")
    p.add_argument("--data", default="data/finewiki")
    p.add_argument("--device", default="cuda")
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--batches", type=int, default=100)
    p.add_argument("--split", choices=("validation", "test"), default="test")
    p.add_argument("--compare-prefill", action="store_true")
    p = commands.add_parser("generate")
    p.add_argument("checkpoint")
    p.add_argument("--tokenizer", default="data/finewiki/tokenizer.json")
    p.add_argument("--prompt", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--bounded", action="store_true")
    p = commands.add_parser("train-draft")
    p.add_argument("checkpoint")
    p.add_argument("--data", default="data/finewiki")
    p.add_argument("--run", default="runs/dspark")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tokens", type=int, default=1_000_000)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--resume")
    for name in ("profile-draft", "generate-draft"):
        p = commands.add_parser(name)
        p.add_argument("draft")
        p.add_argument("--tokenizer", default="data/finewiki/tokenizer.json")
        p.add_argument("--prompt", required=True)
        p.add_argument("--device", default="cuda")
        if name == "profile-draft":
            p.add_argument("--rounds", type=int, default=5)
            p.add_argument("--output", default="runs/dspark/profile.json")
        else:
            p.add_argument("--max-new-tokens", type=int, default=128)
            p.add_argument("--temperature", type=float, default=0.8)
            p.add_argument("--profile", help="Measured timing curves for confidence scheduling")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "params":
        c = config_for(args)
        # Meta allocation reports the full configuration without allocating its weights.
        with torch.device("meta"):
            model = MiniDeepSeek(c)
        write_result(model.parameter_report(), args.output)
        if args.save_config:
            c.save(args.save_config)
    elif args.command == "prepare":
        if not args.encode_only:
            write_result(
                collect(args.data, args.source, args.documents, args.revision, args.preview)
            )
            tokenizer = train_tokenizer(args.data, args.vocab_size)
        else:
            tokenizer = Tokenizer.from_file(str(Path(args.data) / "tokenizer.json"))
        write_result(encode_corpus(args.data, tokenizer))
    elif args.command == "benchmark":
        result = benchmark(
            config_for(args),
            args.length,
            args.device,
            args.steps,
            args.seconds,
            args.tokens_per_step,
        )
        write_result(result, args.output)
    elif args.command == "train":
        train(
            args.data,
            args.run,
            config_for(args),
            args.tokens,
            args.device,
            args.length,
            args.tokens_per_step,
            args.resume,
            args.max_steps,
            args.lr,
            args.initialize,
            args.replay,
            args.fake_quant,
        )
    elif args.command == "train-draft":
        from .dspark import train_draft

        print(
            train_draft(
                args.checkpoint,
                args.data,
                args.run,
                args.device,
                args.tokens,
                args.length,
                args.max_steps,
                args.width,
                args.resume,
            )
        )
    elif args.command in ("profile-draft", "generate-draft"):
        from .dspark import DSpark, profile_draft, speculative_round

        saved = torch.load(args.draft, map_location="cpu", weights_only=False)
        if fingerprint(saved["backbone"]) != saved["backbone_sha256"]:
            raise ValueError("Draft backbone checkpoint has changed")
        backbone, state = load_checkpoint(saved["backbone"], args.device)
        if fingerprint(args.tokenizer) != state["metadata"]["tokenizer_sha256"]:
            raise ValueError("Tokenizer mismatch")
        draft = DSpark(backbone, saved["width"], saved["markov_rank"], saved["positions"]).to(
            args.device
        )
        draft.load_state_dict(saved["draft"])
        backbone.eval()
        draft.eval()
        tokenizer = Tokenizer.from_file(args.tokenizer)
        ids = [backbone.config.bos_id] + tokenizer.encode(args.prompt).ids
        if len(ids) + draft.positions > backbone.config.max_seq_len:
            raise ValueError("Prompt is too long for draft verification")
        if args.command == "profile-draft":
            write_result(
                profile_draft(
                    backbone, draft, torch.tensor([ids], device=args.device), args.rounds
                ),
                args.output,
            )
        else:
            if (
                args.temperature <= 0
                or len(ids) + args.max_new_tokens > backbone.config.max_seq_len
            ):
                raise ValueError(
                    "Draft generation requires temperature > 0 and enough context space"
                )
            profile = json.loads(Path(args.profile).read_text()) if args.profile else None
            generated = 0
            done = False
            while generated < args.max_new_tokens and not done:
                remaining = args.max_new_tokens - generated
                # Bound the last round before proposing; the scheduler may otherwise choose more.
                round_profile = profile if remaining >= draft.positions else None
                tokens, _, _ = speculative_round(
                    backbone,
                    draft,
                    torch.tensor([ids], device=args.device),
                    temperature=args.temperature,
                    verification_length=min(draft.positions, remaining),
                    profile=round_profile,
                )
                for token in tokens:
                    ids.append(token)
                    generated += 1
                    if token == backbone.config.eos_id:
                        done = True
                        break
            print(tokenizer.decode(ids), flush=True)
    elif args.command == "evaluate":
        model, state = load_checkpoint(args.checkpoint, args.device)
        from .data import verify_corpus

        encoded = verify_corpus(args.data)
        if state["metadata"]["tokenizer_sha256"] != encoded["tokenizer_sha256"]:
            raise ValueError("Tokenizer fingerprint differs from training")
        operation = compare_prefill if args.compare_prefill else evaluate
        write_result(
            operation(
                model, TokenStream(args.data, args.split, args.length), args.batches, args.device
            )
        )
    elif args.command == "generate":
        model, state = load_checkpoint(args.checkpoint, args.device)
        if fingerprint(args.tokenizer) != state["metadata"]["tokenizer_sha256"]:
            raise ValueError("Tokenizer fingerprint differs from training")
        model.eval()
        tokenizer = Tokenizer.from_file(args.tokenizer)
        ids = [model.config.bos_id] + tokenizer.encode(args.prompt).ids
        if len(ids) + args.max_new_tokens > model.config.max_seq_len:
            raise ValueError("Prompt plus requested output exceeds the context limit")
        logits, cache = model.prefill(torch.tensor([ids], device=args.device), args.bounded)
        for _ in range(args.max_new_tokens):
            token = (
                logits.argmax(-1)
                if args.temperature <= 0
                else torch.multinomial((logits / args.temperature).softmax(-1), 1).squeeze(-1)
            )
            if token.item() == model.config.eos_id:
                break
            ids.append(token.item())
            logits, cache = model.decode_step(token[:, None], cache)
        print(tokenizer.decode(ids), flush=True)


if __name__ == "__main__":
    main()
