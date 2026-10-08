#!/usr/bin/env python3
"""Select an experiment and print its evaluation command; add --run to execute.

Requires PyYAML to print commands. Evaluation also requires this repository's
lm_eval installation and the chosen model backend.
"""

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "lm_eval/tasks/unhelpful_thoughts"
DATASET_ID = "soheeyang/UnhelpfulThoughts"


class ConfigLoader(yaml.SafeLoader):
    """Inspect function references without importing or executing them."""


ConfigLoader.add_constructor("!function", lambda loader, node: loader.construct_scalar(node))


def load_config(path):
    return yaml.load(path.read_text(encoding="utf-8"), Loader=ConfigLoader)


def supported_configs():
    configs = {}
    for path in sorted(TASK_ROOT.rglob("*.yaml")):
        config = load_config(path)
        if config.get("dataset_path") != DATASET_ID:
            continue
        task = config["task"]
        if task in configs:
            raise ValueError(f"Duplicate supported task {task}: {configs[task][0]} and {path}")
        configs[task] = (path, config)
    return configs


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["aime24", "arc", "gpqa", "humaneval", "math500", "harmbench"], default="math500")
    parser.add_argument("--condition", default="baseline", help="Experiment to run, e.g. --dataset math500 --condition recover_irrelevant_10p_r1; use --list to see task IDs")
    parser.add_argument("--list", action="store_true", help="List supported task IDs for --dataset without loading models or datasets")
    parser.add_argument("--model", help="Model repository/path, required unless --list or --model-args supplies pretrained")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--model-args", default="", help="Additional harness model_args, e.g. dtype=bfloat16,tensor_parallel_size=2")
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-gen-toks", type=int, default=1024, help="Maximum generated tokens per response (default: 1024)")
    parser.add_argument("--limit", type=int, default=1, help="Default is a one-example smoke run; use --full for all examples")
    parser.add_argument("--full", action="store_true", help="Evaluate all examples instead of the smoke-run limit")
    parser.add_argument("--output", type=Path, default=Path("results/paper-smoke"))
    parser.add_argument("--seed", type=int, default=42, help="Seed for harness RNGs and vLLM (unless model-args sets its seed explicitly)")
    parser.add_argument("--prompt-suffix", default=None, help="Explicit override of the selected YAML's suffix; preserve newlines in shell quoting")
    parser.add_argument("--temperature", type=float, default=None, help="Override YAML sampling temperature (normally 0.6)")
    parser.add_argument("--confirm-run-unsafe-code", action="store_true", help="Required for HumanEval code execution in an appropriately isolated environment")
    parser.add_argument("--run", action="store_true", help="Execute evaluation; without this flag only print a plan")
    return parser


def build_command(args, config, task_directory):
    model_args = args.model_args
    if args.model:
        if any(part.startswith("pretrained=") for part in model_args.split(",")):
            raise ValueError("Supply pretrained through --model or --model-args, not both")
        model_args = f"pretrained={args.model}" + (f",{model_args}" if model_args else "")
    elif not any(part.startswith("pretrained=") for part in model_args.split(",")):
        raise ValueError("Pass --model or pretrained=... in --model-args")
    if args.backend == "vllm" and not any(part.startswith("seed=") for part in model_args.split(",")):
        model_args += f",seed={args.seed}"
    if args.max_gen_toks < 1 or (not args.full and args.limit < 1):
        raise ValueError("--max-gen-toks and --limit must be positive")
    gen_kwargs = f"max_gen_toks={args.max_gen_toks}"
    if args.temperature is not None:
        if args.temperature < 0:
            raise ValueError("--temperature must be nonnegative")
        gen_kwargs += f",temperature={args.temperature}"
    command = [sys.executable, "-m", "lm_eval", "--model", args.backend,
               "--model_args", model_args, "--tasks", str(task_directory),
               "--batch_size", str(args.batch_size), "--apply_chat_template",
               "--gen_kwargs", gen_kwargs,
               "--log_samples", "--output_path", str(args.output.resolve()),
               "--seed", str(args.seed)]
    # The condition YAML supplies the complete assistant/thought prefix.
    if config.get("prompt_suffix"):
        command.append("--no_generation_prompt")
    if not args.full:
        command.extend(["--limit", str(args.limit)])
    if args.device:
        command.extend(["--device", args.device])
    if args.confirm_run_unsafe_code:
        command.append("--confirm_run_unsafe_code")
    return command


def main():
    parser = make_parser()
    args = parser.parse_args()
    try:
        configs = supported_configs()
        configs = {name: pair for name, pair in configs.items() if pair[0].relative_to(TASK_ROOT).parts[0] == args.dataset}
        if args.list:
            for name, (path, _) in configs.items():
                print(f"{name}\t{path.relative_to(ROOT)}")
            return 0
        condition = args.condition
        name = condition if "__" in condition else f"{args.dataset}__{condition}"
        if name not in configs:
            raise ValueError(f"Unsupported task {name}; use --dataset {args.dataset} --list")
        path, config = configs[name]
        config = dict(config)
        if args.prompt_suffix is not None:
            config["prompt_suffix"] = args.prompt_suffix
        command = build_command(args, config, "<temporary-task-directory>")
        print(f"Task: {name}")
        print(f"Config: {path.relative_to(ROOT)}")
        split = config.get("test_split") or config.get("validation_split")
        print(f"Dataset: {DATASET_ID} / {config['dataset_name']} / {split}")
        if args.prompt_suffix is not None:
            print(f"Prompt suffix override: {args.prompt_suffix!r}")
        print("Mode: " + ("full evaluation" if args.full else f"smoke run ({args.limit} example(s))"))
        if args.dataset == "harmbench":
            print("Use scripts/judge_jailbreak.py to score the saved responses with o4-mini.")
        print(shlex.join(command))
        if not args.run:
            print("Dry run only. Add --run to execute this plan.")
            return 0
        if config.get("unsafe_code") and not args.confirm_run_unsafe_code:
            raise ValueError("HumanEval executes generated code. Use an isolated environment and pass --confirm-run-unsafe-code")
        # Load only this configuration and its helper functions.
        with tempfile.TemporaryDirectory(prefix="unhelpful-task-") as temp:
            task_directory = Path(temp)
            source = path.read_text(encoding="utf-8")
            if args.prompt_suffix is not None:
                source, count = re.subn(r"(?m)^prompt_suffix:.*$", lambda match: "prompt_suffix: " + json.dumps(args.prompt_suffix, ensure_ascii=False), source)
                if count != 1:
                    raise ValueError("Expected exactly one single-line prompt_suffix in selected YAML")
            (task_directory / path.name).write_text(source, encoding="utf-8")
            for helper in path.parent.glob("*.py"):
                shutil.copy2(helper, task_directory / helper.name)
            args.output.mkdir(parents=True, exist_ok=True)
            effective_config = args.output / f"{name}.config.yaml"
            effective_config.write_text(source, encoding="utf-8")
            manifest = {
                "task": name,
                "source_config": str(path.relative_to(ROOT)),
                "source_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "effective_config_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "effective_config": effective_config.name,
                "model": args.model,
                "backend": args.backend,
                "command_template": command,
                "scope": "full" if args.full else "smoke",
            }
            # Model args can contain credentials; do not copy them into a run manifest.
            manifest["command_template"] = ["<model-args>" if i and command[i - 1] == "--model_args" else value for i, value in enumerate(command)]
            (args.output / f"{name}.launch.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            return subprocess.run(build_command(args, config, task_directory), cwd=ROOT).returncode
    except (ValueError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
