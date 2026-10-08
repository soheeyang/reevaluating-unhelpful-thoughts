#!/usr/bin/env python3
"""Offline checks of paper config schemas, request suffixes, and answer filters.

Install PyYAML and Jinja2. No models, datasets, GPU packages, or network calls are
loaded. Selected source functions are compiled with tiny dependency stubs; this
does not validate an installed harness or reproduce numerical paper results.
"""

import ast
import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

from jinja2 import Environment, StrictUndefined, meta

from run_paper_eval import DATASET_ID, ROOT, build_command, make_parser, supported_configs


def check_schema(configs):
    schema = json.loads((ROOT / "docs/dataset_schema.json").read_text(encoding="utf-8"))
    assert schema["dataset_id"] == DATASET_ID
    environment = Environment(undefined=StrictUndefined)
    checked = 0
    for name, (path, config) in configs.items():
        dataset_name = config["dataset_name"]
        dataset = schema["configs"][dataset_name]
        split = config.get("test_split") or config.get("validation_split")
        assert split in dataset["splits"], f"{path}: missing split {split}"
        columns = set(dataset["splits"][split]["columns"])
        if config.get("process_docs"):
            assert dataset_name in {"gpqa", "gpqa_correct"}, f"Add explicit derived-column handling for {path}"
            columns.update({"choice1", "choice2", "choice3", "choice4", "choices", "answer"})
        for field in ["doc_to_text", "doc_to_target", "prompt_suffix"]:
            template = config.get(field)
            if not isinstance(template, str):
                continue
            variables = meta.find_undeclared_variables(environment.parse(template))
            missing = variables - columns
            assert not missing, f"{name}: {field} uses unavailable fields {sorted(missing)}"
            checked += 1
        assert config.get("output_type") == "generate_until", path
    return checked


def compile_source_nodes(path, names, namespace, parent=None):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = tree.body
    if parent:
        nodes = next(node.body for node in nodes if isinstance(node, ast.ClassDef) and node.name == parent)
    selected = [node for node in nodes if getattr(node, "name", None) in names]
    assert len(selected) == len(names), f"Missing source definitions: {names}"
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def check_filters():
    namespace = {"re": re, "Filter": object, "register_filter": lambda name: lambda cls: cls}
    compile_source_nodes(ROOT / "lm_eval/filters/custom.py", {"BoxedFilter", "McqaFilter", "McqaFilterLast"}, namespace)
    boxed = namespace["BoxedFilter"]()
    assert boxed.apply([[r"Reasoning." + "\n\n" + r"Final: \boxed{\frac{1}{2}}"]], [{}]) == [[r"\frac{1}{2}"]]
    assert boxed.apply([[r"Final: \boxed{42", ""]], [{}]) == [["[invalid]", "[invalid]"]]
    assert namespace["McqaFilterLast"]().apply([[r"Earlier \boxed{no}. Revised \boxed{yes}."]], [{}]) == [["yes"]]
    assert namespace["McqaFilter"]().apply([["Thinking\n\nThe answer is C."]], [{}]) == [["C"]]
    return 4


def check_request_suffix(configs):
    namespace = {"deepcopy": copy.deepcopy, "Instance": SimpleNamespace}
    compile_source_nodes(ROOT / "lm_eval/api/task.py", {"construct_requests"}, namespace, parent="ConfigurableTask")
    fake = SimpleNamespace(OUTPUT_TYPE="generate_until", config=SimpleNamespace(generation_kwargs={"temperature": 0.6}, doc_to_image=None))
    construct = namespace["construct_requests"]
    context = "<user>A synthetic problem</user>"
    no_suffix = construct(fake, {"problem": "x"}, context, prompt_suffix="")
    assert no_suffix.arguments[0] == context
    suffix_template = configs["math500__recover_irrelevant_r1"][1]["prompt_suffix"]
    suffix = Environment(undefined=StrictUndefined).from_string(suffix_template).render(irrelevant_thought="SYNTHETIC THOUGHT")
    request = construct(fake, {"problem": "x"}, context, prompt_suffix=suffix)
    assert request.arguments[0] == context + "<｜Assistant｜><think>\nSYNTHETIC THOUGHT"
    request.arguments[1]["temperature"] = 0
    assert fake.config.generation_kwargs["temperature"] == 0.6
    return 3


def check_launcher(configs):
    args = make_parser().parse_args(["--model", "example/model"])
    baseline = configs["math500__baseline"][1]
    injected = configs["math500__recover_irrelevant_r1"][1]
    baseline_cmd = build_command(args, baseline, Path("synthetic-config-directory"))
    injected_cmd = build_command(args, injected, Path("synthetic-config-directory"))
    assert "--apply_chat_template" in baseline_cmd
    assert "--no_generation_prompt" not in baseline_cmd
    assert "--no_generation_prompt" in injected_cmd
    assert "--limit" in injected_cmd and not args.run
    args.full = True
    assert "--limit" not in build_command(args, injected, Path("synthetic-config-directory"))
    return 5


def check_experiment_prompts(configs):
    environment = Environment(undefined=StrictUndefined)
    identification_prefix = "<｜Assistant｜><think>\nOkay, so I need to figure out whether the thinking process provided"
    checked = 0
    for name, (_, config) in configs.items():
        if "__attack_in_input_" in name:
            prompt = environment.from_string(config["doc_to_text"]).render(
                question="FORMATTED QUESTION", unformatted_question="RAW QUESTION",
            )
            assert prompt.startswith("FORMATTED QUESTION "), name
            assert "{question}" not in prompt, name
            assert prompt.endswith("RAW QUESTION"), name
            checked += 1
        elif "__identify_" in name:
            assert config["prompt_suffix"] == identification_prefix, name
            command = build_command(make_parser().parse_args(["--model", "example/model"]), config, Path("synthetic-config-directory"))
            assert "--no_generation_prompt" in command, name
            checked += 1
    return checked


def main():
    configs = supported_configs()
    assert configs, "No supported configs found"
    print(f"PASS: {len(configs)} supported configs have unique task IDs")
    templates = check_schema(configs)
    print(f"PASS: {templates} Jinja templates use fields available in the recorded dataset schema")
    checks = check_filters() + check_request_suffix(configs) + check_launcher(configs) + check_experiment_prompts(configs)
    print(f"PASS: {checks} source-level filter, request, and launcher checks")
    print("Scope: offline source/config validation only; no model inference or paper-result reproduction.")


if __name__ == "__main__":
    main()
