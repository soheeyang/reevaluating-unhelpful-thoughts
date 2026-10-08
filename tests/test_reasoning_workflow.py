"""Notebook parity and complete, offline sample-log scoring checks."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import reasoning_answers as answers
from scripts import score_reasoning as workflow


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "METACOG_remaster_final.ipynb"
pytestmark = pytest.mark.filterwarnings("ignore:invalid escape sequence:SyntaxWarning")


@pytest.mark.parametrize("task,response,target,expected", [
    ("aime24", r"First 12, finally \boxed{007}", "7", 1),
    ("aime24", "The answer: 42", "042", 1),
    ("aime24", r"\boxed{42", "42", 0),
    ("math500", r"\boxed{\frac{-1}{-2}}", "0.5", 1),
    ("math500", r"\boxed{\dfrac12}", "0.5", 1),
    ("math500", r"\boxed{\sqrt{2}}", r"\sqrt2", 1),
    ("math500", r"\boxed{3\text{ inches}}", "3", 1),
    ("math500", r"\boxed{2x}", "x+x", 0),
    ("arc", r"\boxed{A} and \boxed{C}", "c", 1),
    ("gpqa", "Answer: d", "D", 1),
    ("identify_arc", r"\boxed{NO}", "no", 1),
    ("identify_math500", "No", "no", 0),
    ("identify_humaneval", r"\boxed{NO}", "no", 0),
    ("humaneval", "unexecuted code", "assert test", 0.25),
])
def test_answer_policy(task, response, target, expected):
    row = dict(task=task, response=response, target=target, prompt="", score=0.25)
    row["gold"] = answers.get_gold(row)
    row["prediction"] = answers.get_prediction(row)
    assert answers.get_score(row) == expected


def test_answer_functions_match_notebook():
    if not NOTEBOOK.exists():
        pytest.skip("Notebook is retained only in the private repository")
    notebook = json.loads(NOTEBOOK.read_text())
    functions = {node.name: node for node in ast.parse(Path(answers.__file__).read_text()).body
                 if isinstance(node, ast.FunctionDef)}
    checked = []
    for index in (14, 18, 19, 22):
        for node in ast.parse("".join(notebook["cells"][index]["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name in functions and node.name != "evaluate_fraction":
                assert ast.dump(node) == ast.dump(functions[node.name]), node.name
                checked.append(node.name)
    assert len(checked) >= 20


def test_generated_fraction_cannot_execute_python(tmp_path):
    marker = tmp_path / "executed"
    payload = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    expression = r"\frac{" + payload + "}{1}"
    assert answers.evaluate_fraction(expression) == expression
    assert not marker.exists()


def write_run(tmp_path, samples, model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
    date = "2026-10-08T00-00-00.000001"
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"results_{date}.json").write_text(json.dumps({"model_name": model, "configs": dict.fromkeys(samples, {})}))
    paths = []
    for task, rows in samples.items():
        path = tmp_path / f"samples_{task}_{date}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths.append(path)
    return paths


def sample(index, response, target="1"):
    return {"doc_id": index, "doc_hash": f"hash-{index}", "target": target,
            "arguments": {"gen_args_0": {"arg_0": "Question\n<｜Assistant｜><think>"}},
            "resps": [["reasoning</think>" + response]], "exact_match": 0}


def test_end_to_end_scoring_and_macro_is_not_pooled(tmp_path):
    source = tmp_path / "run"
    write_run(source, {
        "math500__baseline": [sample(0, r"\boxed{1}"), sample(1, r"\boxed{2}"), sample(2, r"\boxed{2}")],
        "arc__baseline": [sample(0, r"\boxed{A}", "A")],
    })
    output = tmp_path / "scored"
    workflow.main([str(source), "--output", str(output)])
    samples = pd.read_csv(output / "samples.csv")
    assert samples.score.tolist() == [1, 1, 0, 0]
    assert samples.input_score.tolist() == [0, 0, 0, 0]
    assert samples.doc_hash.tolist() == ["hash-0", "hash-0", "hash-1", "hash-2"]
    summary = pd.read_csv(output / "summary.csv")
    macro = summary[summary.task == "average"].iloc[0]
    assert macro.score == pytest.approx((1 + 1 / 3) / 2)
    assert macro.n_rows == 4 and macro.n_tasks == 2
    assert samples.score.mean() != macro.score


def test_bootstrap_matches_notebook(tmp_path):
    if not NOTEBOOK.exists():
        pytest.skip("Notebook is retained only in the private repository")
    notebook = json.loads(NOTEBOOK.read_text())
    namespace = {"np": np, "pd": pd}
    for index in (36, 39):
        functions = [node for node in ast.parse("".join(notebook["cells"][index]["source"])).body
                     if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    frame = pd.DataFrame([dict(experiment="baseline", model="R1", task=task, score=score)
                          for task, score in [("math500", 1), ("math500", 0), ("arc", 1)]])
    state = np.random.get_state()
    try:
        np.random.seed(42)
        expected = namespace["add_confidence_intervals"](frame)
        expected = namespace["add_task_averages_with_ci"](frame, expected)
    finally:
        np.random.set_state(state)
    actual = workflow.aggregate_scores(frame, seed=42)
    pd.testing.assert_frame_equal(actual[list(expected.columns)], expected)


def test_humaneval_carries_harness_pass_at_one(tmp_path):
    row = sample(0, "unexecuted code", "assert test")
    paths = write_run(tmp_path, {"humaneval__baseline": [row]})
    with pytest.raises(ValueError, match="pass@1"):
        workflow.score_file(paths[0])
    row["pass@1"] = 0.25
    paths = write_run(tmp_path, {"humaneval__baseline": [row]})
    assert workflow.score_file(paths[0]).score.tolist() == [0.25]


def test_incorrect_selection_and_matched_baseline(tmp_path):
    source = tmp_path / "run"
    write_run(source, {
        "math500__baseline": [sample(0, r"\boxed{1}"), sample(1, r"\boxed{2}")],
        "math500__recover_incorrect_r1": [sample(0, r"\boxed{1}"), sample(1, r"\boxed{1}")],
    })
    construction = tmp_path / "construction.csv"
    pd.DataFrame([
        dict(model="DeepSeek-1.5B", task="math500", index=0, score=1),
        dict(model="DeepSeek-1.5B", task="math500", index=1, score=0),
    ]).to_csv(construction, index=False)
    output = tmp_path / "scored"
    workflow.main([str(source), "--output", str(output), "--construction-input", str(construction)])
    frame = pd.read_csv(output / "samples.csv")
    assert frame.loc[frame.experiment == "recover_incorrect_r1", "index"].tolist() == [1]
    assert frame.loc[frame.experiment == "baseline_incorrect", "index"].tolist() == [1]
    assert frame.loc[frame.experiment == "baseline_incorrect", "score"].tolist() == [0]
    assert json.loads((output / "incorrect_indices.json").read_text()) == {"math500": ["1"]}


def test_missing_subset_and_duplicate_runs_fail(tmp_path, capsys):
    paths = write_run(tmp_path / "run", {"math500__recover_incorrect_r1": [sample(0, r"\boxed{1}")]})
    with pytest.raises(SystemExit):
        workflow.main([str(paths[0]), "--output", str(tmp_path / "scored")])
    assert "need --incorrect-indices" in capsys.readouterr().err
    assert not (tmp_path / "scored").exists()
    with pytest.raises(SystemExit):
        workflow.main([str(paths[0]), str(paths[0]), "--output", str(tmp_path / "scored")])
    assert "nonoverlapping" in capsys.readouterr().err


def test_old_names_and_metadata_conflicts_fail(tmp_path):
    paths = write_run(tmp_path, {"math500__baseline": [sample(0, "1")]})
    with pytest.raises(ValueError, match="disagrees"):
        workflow.score_file(paths[0], model="different-model")
    with pytest.raises(ValueError, match="disagrees"):
        workflow.score_file(paths[0], task_id="arc__baseline")
    standalone = tmp_path / "renamed.jsonl"
    standalone.write_text(paths[0].read_text())
    for task in ("math500__revert_irrelevant_r1", "math500__recover_hard_irrelevant_r1"):
        with pytest.raises(ValueError, match="Unsupported"):
            workflow.score_file(standalone, task_id=task, model="R1")


@pytest.mark.parametrize("baseline_ids,incorrect_ids", [([0], [1]), ([0], [0, 1]), ([0, 1], [1])])
def test_incomplete_matched_subset_fails(baseline_ids, incorrect_ids):
    frame = pd.DataFrame([
        dict(model="DeepSeek-7B", task="math500", experiment=condition, index=index, score=1)
        for condition, indices in [("baseline", baseline_ids), ("recover_incorrect_r1", incorrect_ids)]
        for index in indices
    ])
    with pytest.raises(ValueError, match="Incomplete incorrect-question subset"):
        workflow.select_incorrect_subset(frame, {"math500": [0, 1]})


def test_document_identity_disagreement_fails_but_correct_dataset_is_separate():
    frame = pd.DataFrame([
        dict(task="math500", experiment="baseline", index=0, doc_hash="original"),
        dict(task="identify_math500", experiment="identify_incorrect", index=0, doc_hash="changed"),
    ])
    with pytest.raises(ValueError, match="Question mismatch"):
        workflow.validate_sample_identities(frame)
    frame.loc[1, "experiment"] = "identify_correct"
    workflow.validate_sample_identities(frame)


@pytest.mark.parametrize("changes,error", [
    ({"experiment": "recover_irrelevant_r1"}, "baseline answers"),
    ({"task": "identify_math500"}, "benchmark names"),
    ({"score": -1}, "between zero and one"),
    ({"score": float("inf")}, "between zero and one"),
    ({"index": "-1"}, "nonnegative integers"),
    ({"index": "0.5"}, "nonnegative integers"),
])
def test_construction_input_rejects_other_populations_and_invalid_values(tmp_path, changes, error):
    row = dict(model="DeepSeek-1.5B", task="math500", experiment="baseline_incorrect", index=0, score=0)
    row.update(changes)
    path = tmp_path / "construction.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match=error):
        workflow.incorrect_indices_from_scores(path)
