#!/usr/bin/env python3
"""Score reasoning sample logs and write accuracy summaries with bootstrap intervals."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts import reasoning_answers as answers
except ModuleNotFoundError:
    import reasoning_answers as answers


TASKS = {"aime24", "arc", "gpqa", "humaneval", "math500"}
TASK_ROOT = Path(__file__).resolve().parents[1] / "lm_eval/tasks/unhelpful_thoughts"


def sample_prompt(row):
    arguments = row.get("arguments")
    if isinstance(arguments, dict):
        argument = arguments.get("gen_args_0", arguments)
        if isinstance(argument, dict) and isinstance(argument.get("arg_0"), str):
            return argument["arg_0"]
    if isinstance(arguments, list) and arguments:
        first = arguments[0]
        if isinstance(first, str):
            return first
        if isinstance(first, list) and first and isinstance(first[0], str):
            return first[0]
    raise ValueError("Sample is missing its generation prompt")


def sample_metadata(path, *, task_id=None, model=None):
    """Read the task and matching run metadata from harness output names."""
    match = re.fullmatch(r"samples_(.+)_(\d{4}-\d{2}-\d{2}T.+)\.jsonl", path.name)
    result = {}
    if match:
        if task_id is not None and task_id != match[1]:
            raise ValueError(f"--task {task_id} disagrees with the sample filename {path.name}")
        task_id = match[1]
        result_path = path.with_name(f"results_{match[2]}.json")
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if task_id not in result.get("configs", {}):
                raise ValueError(f"{task_id} is absent from {result_path}")
    if not task_id or "__" not in task_id:
        raise ValueError(f"Cannot identify the task in {path}; pass --task, e.g. math500__baseline")
    benchmark, condition = task_id.split("__", 1)
    config_exists = (TASK_ROOT / benchmark / "baseline.yaml").exists() if condition == "baseline" else bool(list((TASK_ROOT / benchmark).rglob(f"{task_id}.yaml")))
    if benchmark not in TASKS or not config_exists:
        raise ValueError(f"Unsupported reasoning task: {task_id}")
    logged_model = result.get("model_name")
    if not logged_model:
        model_args = result.get("config", {}).get("model_args", {})
        if isinstance(model_args, dict):
            logged_model = model_args.get("pretrained")
        elif isinstance(model_args, str):
            logged_model = next((part.split("=", 1)[1] for part in model_args.split(",")
                                 if part.startswith("pretrained=")), None)
    if model and logged_model and model != logged_model:
        raise ValueError(f"--model {model} disagrees with the logged model {logged_model}")
    model = model or logged_model
    if not isinstance(model, str) or not model:
        raise ValueError(f"Cannot identify the model in {path}; pass --model")
    return benchmark, condition, model


def score_file(path, *, task_id=None, model=None):
    benchmark, condition, model = sample_metadata(path, task_id=task_id, model=model)
    task = f"identify_{benchmark}" if condition.startswith("identify_") else benchmark
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        index = raw.get("doc_id")
        if type(index) is not int or index < 0:
            raise ValueError(f"{path}:{line_number}: doc_id must be a nonnegative integer")
        generation = raw.get("resps")
        while isinstance(generation, list) and len(generation) == 1:
            generation = generation[0]
        if not isinstance(generation, str):
            raise ValueError(f"{path}:{line_number}: expected exactly one generation")
        target = raw.get("target")
        if type(target) not in (str, int, float):
            raise ValueError(f"{path}:{line_number}: expected a scalar target")
        prompt = sample_prompt(raw)
        row = {
            "experiment": condition, "model": model, "task": task, "index": index,
            "doc_hash": raw.get("doc_hash", ""), "source": str(path),
            "prompt": prompt, "generation": generation,
            "response": answers.extract_response(prompt + generation), "target": str(target),
            "input_score": raw.get("pass@1") if task == "humaneval" else raw.get("exact_match"),
        }
        if task == "humaneval":
            value = row["input_score"]
            if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{path}:{line_number}: HumanEval requires a valid harness pass@1")
            row["score"] = value
        row["gold"] = answers.get_gold(row)
        kwargs = {"similarity_match": True} if "1.5B" in model and task in {"arc", "gpqa"} else {}
        row["prediction"] = answers.get_prediction(row, **kwargs)
        row["score"] = float(answers.get_score(row))
        rows.append(row)
    if not rows:
        raise ValueError(f"No samples in {path}")
    return pd.DataFrame(rows)


def incorrect_indices_from_scores(path):
    """Select questions answered incorrectly by the 1.5B construction model."""
    frame = pd.read_csv(path, dtype={"index": str}, keep_default_na=False)
    if not {"model", "task", "index", "score"} <= set(frame):
        raise ValueError("Construction scores need model, task, index and score columns")
    frame = frame.loc[frame.model.str.contains("1.5B", regex=False)].copy()
    if "experiment" in frame and not frame.experiment.isin(["baseline", "baseline_incorrect"]).all():
        raise ValueError("Construction input must contain 1.5B baseline answers; remove identification and recovery runs")
    if not frame.task.isin(TASKS).all():
        raise ValueError(f"Construction tasks must use benchmark names: {', '.join(sorted(TASKS))}")
    frame["score"] = pd.to_numeric(frame.score, errors="raise")
    if not np.isfinite(frame.score).all() or not frame.score.between(0, 1).all():
        raise ValueError("Construction scores must be finite values between zero and one")
    validate_incorrect_indices({task: group["index"].tolist() for task, group in frame.groupby("task")})
    selected = frame.loc[frame.score == 0]
    if selected.empty:
        raise ValueError("Construction scores contain no incorrect 1.5B answers")
    return {task: set(group["index"]) for task, group in selected.groupby("task")}


def validate_incorrect_indices(indices):
    if not isinstance(indices, dict) or not indices or set(indices) - TASKS:
        raise ValueError(f"Incorrect indices must map benchmark names ({', '.join(sorted(TASKS))}) to row indices")
    for task, values in indices.items():
        if not isinstance(values, (list, set)) or not values:
            raise ValueError(f"Incorrect indices for {task} must be a nonempty list")
        if any(type(index) not in (str, int) or re.fullmatch(r"0|[1-9][0-9]*", str(index)) is None for index in values):
            raise ValueError(f"Incorrect indices for {task} must be nonnegative integers")


def select_incorrect_subset(frame, indices):
    """Filter incorrect-thought runs and add baselines on the same questions."""
    validate_incorrect_indices(indices)
    indices = {task: {str(index) for index in values} for task, values in indices.items()}
    indices.update({"identify_" + task: values for task, values in list(indices.items())
                    if not task.startswith("identify_")})
    is_incorrect = frame.experiment.str.contains("incorrect", regex=False)
    missing = set(frame.loc[is_incorrect, "task"]) - set(indices)
    if missing:
        raise ValueError(f"Incorrect-question indices are missing tasks: {sorted(missing)}")
    selected = frame.loc[[not incorrect or str(index) in indices[task]
                         for incorrect, index, task in zip(is_incorrect, frame["index"], frame.task)]].copy()

    def require_coverage(group, expected, name):
        actual = set(group["index"].astype(str))
        if actual != expected:
            missing = sorted(expected - actual)
            raise ValueError(f"Incomplete incorrect-question subset for {name}: missing indices {missing[:10]}. "
                             "Include the full sample log for this run; all compared runs must cover the supplied indices.")

    for (model, experiment, task), group in frame.loc[is_incorrect].groupby(["model", "experiment", "task"]):
        retained = group.loc[group["index"].astype(str).isin(indices[task])]
        require_coverage(retained, indices[task], f"{model}/{task}/{experiment}")
    baselines = []
    for task, values in indices.items():
        baselines_for_task = selected[(selected.experiment == "baseline") & (selected.task == task)
                                      & selected.model.str.contains("DeepSeek", regex=False)]
        for model, group in baselines_for_task.groupby("model"):
            baseline = group.loc[group["index"].astype(str).isin(values)].copy()
            require_coverage(baseline, values, f"{model}/{task}/baseline")
            baseline["experiment"] = "baseline_incorrect"
            baselines.append(baseline)
    return pd.concat([selected, *baselines], ignore_index=True)


def validate_sample_identities(frame):
    """Compare document hashes when multiple runs use the same question index."""
    standard = frame.loc[~frame.experiment.str.startswith("identify_correct")].copy()
    standard["benchmark"] = standard.task.str.replace(r"^identify_", "", regex=True)
    for (benchmark, index), group in standard.groupby(["benchmark", "index"]):
        hashes = {value for value in group.doc_hash if isinstance(value, str) and value}
        if len(hashes) > 1:
            raise ValueError(f"Question mismatch for {benchmark} index {index}: document hashes differ across runs. "
                             "Use logs evaluated on the same questions in the same order.")


def aggregate_scores(frame, *, n_bootstraps=1000, confidence=0.95, seed=42):
    """Per-task means and an equally weighted mean across tasks; percentile CIs."""
    if frame.empty or n_bootstraps < 1 or not 0 < confidence < 1:
        raise ValueError("Need samples, positive bootstrap count and confidence between zero and one")
    rng = np.random.RandomState(seed)
    percentiles = [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100]
    output = []
    for (experiment, model, task), group in frame.groupby(["experiment", "model", "task"]):
        values = group.score.to_numpy(dtype=float)
        means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_bootstraps)]
        lower, upper = np.percentile(means, percentiles)
        output.append(dict(experiment=experiment, model=model, task=task, score=float(np.mean(values)),
                           ci_lower=lower, ci_upper=upper, n_rows=len(values), n_tasks=1))
    for (experiment, model), group in frame.groupby(["experiment", "model"]):
        tasks = group.task.unique()
        if len(tasks) < 2:
            continue
        values = [group.loc[group.task == task, "score"].to_numpy(dtype=float) for task in tasks]
        means = [np.mean([np.mean(rng.choice(scores, size=len(scores), replace=True)) for scores in values])
                 for _ in range(n_bootstraps)]
        lower, upper = np.percentile(means, percentiles)
        output.append(dict(experiment=experiment, model=model, task="average",
                           score=float(np.mean([np.mean(scores) for scores in values])),
                           ci_lower=lower, ci_upper=upper, n_rows=len(group), n_tasks=len(tasks)))
    return pd.DataFrame(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Sample JSONL files or evaluation output directories")
    parser.add_argument("--output", type=Path, required=True, help="Directory for samples.csv and summary.csv")
    parser.add_argument("--task", help="Full task ID when scoring a renamed sample file")
    parser.add_argument("--model", help="Model name when matching harness run metadata is unavailable")
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--incorrect-indices", type=Path, help="JSON mapping benchmarks to lists of dataset row indices")
    subset.add_argument("--construction-input", type=Path, help="Scored 1.5B construction samples.csv for incorrect-question selection")
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        paths = []
        for path in args.inputs:
            paths.extend(sorted(path.rglob("samples_*.jsonl")) if path.is_dir() else [path])
        if not paths or len({path.resolve() for path in paths}) != len(paths):
            raise ValueError("Select nonempty, nonoverlapping sample files or directories")
        outputs = [args.output / name for name in ("samples.csv", "summary.csv", "incorrect_indices.json")]
        if any(path.exists() for path in outputs):
            raise ValueError("Output files already exist; choose a new --output directory")
        frame = pd.concat([score_file(path, task_id=args.task, model=args.model) for path in paths], ignore_index=True)
        if frame.duplicated(["model", "experiment", "task", "index"]).any():
            raise ValueError("Duplicate samples: select one run per model, condition and benchmark")
        validate_sample_identities(frame)
        indices = None
        if args.incorrect_indices:
            indices = json.loads(args.incorrect_indices.read_text())
            if not isinstance(indices, dict) or any(not isinstance(values, list) for values in indices.values()):
                raise ValueError("--incorrect-indices must map benchmarks to lists of dataset row indices")
        elif args.construction_input:
            indices = incorrect_indices_from_scores(args.construction_input)
        if indices is None and frame.experiment.str.contains("incorrect", regex=False).any():
            raise ValueError("Incorrect-thought runs need --incorrect-indices or --construction-input to select questions")
        if indices is not None:
            frame = select_incorrect_subset(frame, indices)
        summary = aggregate_scores(frame, n_bootstraps=args.n_bootstraps, confidence=args.confidence, seed=args.seed)
        args.output.mkdir(parents=True, exist_ok=True)
        frame.to_csv(outputs[0], index=False)
        summary.to_csv(outputs[1], index=False)
        if indices is not None:
            outputs[2].write_text(json.dumps({task: sorted(map(str, values)) for task, values in indices.items()}, indent=2) + "\n")
        print(summary.to_string(index=False))
        print(f"Saved {len(frame)} scored samples and {len(summary)} summary rows to {args.output}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
