# How Well Can Reasoning Models Identify and Recover from Unhelpful Thoughts?

Sohee Yang, Sang-Woo Lee, Nora Kassner, Daniela Gottesman, Sebastian Riedel, and Mor Geva. **Findings of EMNLP 2025.**

[Paper](https://aclanthology.org/2025.findings-emnlp.370/) · [Dataset](https://huggingface.co/datasets/soheeyang/UnhelpfulThoughts)

This repository contains code and experiment configurations for testing whether reasoning models can identify unhelpful thoughts and recover after those thoughts are inserted into their reasoning. The experiments cover five reasoning benchmarks and a jailbreak benchmark.

The code extends EleutherAI's [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) with thought injection and answer filters. It includes 210 experiment configurations, a launcher, and scripts for answer scoring and jailbreak judging. See the paper for thought construction and the appendix analyses.

## Quickstart

Use Python 3.12 and a compatible CUDA/PyTorch installation for vLLM. Install this checkout: the experiments use its custom thought-prefix handling, which is absent from an unmodified harness installation.

```bash
git clone https://github.com/soheeyang/reevaluating-unhelpful-thoughts.git
cd reevaluating-unhelpful-thoughts
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[vllm,math,scoring]'
```

Start with one MATH-500 question, an irrelevant thought cut to 10% of its original length, and R1-Distill Qwen-7B:

```bash
python scripts/run_paper_eval.py \
  --dataset math500 --condition recover_irrelevant_10p_r1 \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --backend vllm \
  --model-args tensor_parallel_size=1 \
  --limit 1 --max-gen-toks 128 --output results/smoke
```

The launcher prints the selected task and command. Append `--run` to execute it. This short run checks the evaluation setup; use the settings below for full experiments. Adjust tensor parallelism to your hardware.

To list tasks and check the configuration/schema without installing a model backend:

```bash
python -m pip install PyYAML Jinja2
python scripts/run_paper_eval.py --dataset math500 --list
python scripts/validate_paper_release.py
```

## Data and terminology

All configurations use the public dataset **`soheeyang/UnhelpfulThoughts`**.

| Benchmark | Dataset configuration | Evaluation split | Full questions | Incorrect-thought subset reported in the paper | Correct-thought control |
| --- | --- | --- | ---: | ---: | ---: |
| AIME 2024 | `aime24` | `train` | 30 | 28 | 25 |
| ARC Challenge | `arc` | `test` | 1,172 | 779 | 1,160 |
| GPQA Diamond | `gpqa` | `train` | 198 | 188 | 165 |
| HumanEval | `humaneval` | `test` | 164 | 99 | 160 |
| MATH-500 | `math500` | `test` | 500 | 146 | 485 |
| HarmBench | `harmbench` | `train` | 200 | — | — |

The `train` split name is inherited from the dataset; these are zero-shot evaluations, not training runs. Correct-thought controls use separate configurations such as `math500_correct`, with the same split name. Dataset sizes are from Table 3 of the paper.

The dataset already uses the paper's field names:

| Thought | Dataset field | Recovery condition |
| --- | --- | --- |
| Uninformative | `uninformative_thought` | `recover_uninformative_r1` |
| Irrelevant, 10% | `irrelevant_thought_10p` | `recover_irrelevant_10p_r1` |
| Irrelevant, 100% | `irrelevant_thought` | `recover_irrelevant_r1` |
| Misdirecting, 10% | `misdirecting_thought_10p` | `recover_misdirecting_10p_r1` |
| Misdirecting, 100% | `misdirecting_thought` | `recover_misdirecting_r1` |
| Incorrect | `incorrect_thought` | `recover_incorrect_r1` |
| Correct control | `correct_thought` | Identification only: `identify_correct` |

Jailbreak experiments use `attack_in_input` and `attack_in_thought` for the two attack placements.

A direct dataset load is:

```python
from datasets import load_dataset

data = load_dataset(
    "soheeyang/UnhelpfulThoughts", "math500", split="test",
)
```

## Models and shared settings

Braces below denote separate checkpoints.

| Family | Hugging Face checkpoints | Maximum new tokens |
| --- | --- | ---: |
| R1-Distill | `deepseek-ai/DeepSeek-R1-Distill-Qwen-{7B,14B,32B}`; `deepseek-ai/DeepSeek-R1-Distill-Llama-{8B,70B}` | 32,668 |
| s1.1 | `simplescaling/s1.1-{7B,14B,32B}` | 16,384 |
| EXAONE Deep | `LGAI-EXAONE/EXAONE-Deep-{2.4B,7.8B,32B}` | 16,384 |

Main identification, recovery, and reevaluation-cue experiments use the five R1 models. Cross-family recovery and jailbreak experiments also use the six s1.1/EXAONE models. R1-Distill Qwen-1.5B is used for incorrect-thought construction, not the main evaluation grid.

| Setting | Value |
| --- | --- |
| Backend | vLLM through this modified harness; `--backend hf` is also available for debugging. |
| Prompting | User-message instructions, model chat template, zero-shot. Injected thoughts enter the assistant's reasoning prefix. |
| Sampling | Temperature `0.6`, one generation per example, seed `42`. The launcher sets harness and vLLM seeds unless explicitly overridden. |
| Token limits | Set the family-specific **new-token** budget above. Context length must also accommodate the prompt. |
| Precision | Half precision; the backend defaults to `dtype=auto`. |
| Paper aggregation | Mean accuracy within each dataset, then an unweighted mean across the five datasets. |
| Paper intervals | 95% percentile bootstrap, 1,000 resamples. |

The paper reports using 4–8 H100, L40S, and/or A6000 GPUs. The launcher's default is a one-example run with 1,024 new tokens; `--full` removes the example limit but does **not** change the token budget. Supply both settings explicitly for a full experiment.

## Run the experiments

Choose a benchmark with `--dataset` and an experiment with `--condition`. For example, `--dataset math500 --condition recover_irrelevant_10p_r1` runs short-irrelevant recovery on MATH-500. The corresponding task is named `math500__recover_irrelevant_10p_r1`. Its prompt and settings are defined under [`lm_eval/tasks/unhelpful_thoughts/`](lm_eval/tasks/unhelpful_thoughts/).

### Recovery and baselines

```bash
python scripts/run_paper_eval.py \
  --dataset math500 --condition recover_irrelevant_10p_r1 \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --backend vllm \
  --model-args tensor_parallel_size=1 \
  --seed 42 --full --max-gen-toks 32668 \
  --output results/math500/recover_irrelevant_10p/qwen7b --run
```

Use `--condition baseline` for no injected thought. Repeat over `aime24`, `arc`, `gpqa`, `humaneval`, and `math500`, and the relevant model checkpoints. Use distinct output directories per model and condition.

For the main six conditions, use the recovery names in the thought table above. The length experiment also uses `recover_irrelevant_33p_r1` and `recover_irrelevant_66p_r1`. Percentages refer to **character-prefix truncation followed by trimming at a word boundary**, not tokenizer-token fractions.

For cross-family recovery, use `recover_irrelevant_10p_s1` or `recover_irrelevant_10p_exaone`, select the corresponding checkpoint, and set `--max-gen-toks 16384`. The suffix selects the family's thought delimiters.

HumanEval runs generated Python. Use an isolated execution environment, set `HF_ALLOW_CODE_EVAL=1`, and pass `--confirm-run-unsafe-code`. These options acknowledge execution; they do not create a sandbox.

### Identification

Replace `recover_…_r1` with `identify_…`: for example, `identify_misdirecting_10p`. Use `identify_correct` for the helpful-thought control. The six unhelpful conditions target `no`; the correct-thought control targets `yes`.

Identification configurations start the model's reasoning with “Okay, so I need to figure out whether the thinking process provided”, as described in the paper:

```bash
python scripts/run_paper_eval.py \
  --dataset math500 --condition identify_irrelevant_10p \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --backend vllm \
  --model-args tensor_parallel_size=1 \
  --seed 42 --full --max-gen-toks 32668 \
  --output results/math500/identify_irrelevant_10p/qwen7b --run
```

### Reevaluation cues

For the same six R1 recovery conditions, change `recover_` to:

- `aha_`: append “But wait, let me think again.” after the injected thought.
- `instruct_`: add the explicit instruction to reevaluate the supplied thought.

For example, compare `recover_misdirecting_r1`, `aha_misdirecting_r1`, and `instruct_misdirecting_r1` on the same questions and thoughts. Keep the `_r1` suffix and the R1 generation settings.

### Jailbreak placement

Use `--dataset harmbench` and a complete task ID:

```bash
python scripts/run_paper_eval.py \
  --dataset harmbench \
  --condition jailbreak_benign_intent__attack_in_thought_r1 \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --backend vllm \
  --model-args tensor_parallel_size=1 \
  --seed 42 --full --max-gen-toks 32668 \
  --output results/harmbench/attack_in_thought/benign_intent/qwen7b --run
```

`attack_in_thought` puts a harmful request inside the thought while the user asks a harmless question. `attack_in_input` puts the harmful request and attack text in the user message. Use family suffixes `_r1`, `_s1`, or `_exaone`, with the corresponding checkpoint and token budget.

The ten prompt variants are `benign_intent`, `combined_fiction`, `creative_purpose`, `exam_claim`, `education_purpose`, `full_education_defense`, `fiction_clarification`, `storytelling`, `teacher_verification`, and `full_fiction_defense`. List complete IDs with `python scripts/run_paper_eval.py --dataset harmbench --list`. The design is 200 questions × 10 prompts × 2 placements × 11 models.

After generating responses, score them with the paper's `o4-mini` judge:

```bash
python -m pip install -e '.[judge]'
# Set OPENAI_API_KEY in your environment before running the judge.
python scripts/judge_jailbreak.py \
  --input results/harmbench --output results/harmbench-judged --run
```

The script uses the notebook's prompts to classify input-attack responses as `HARMFUL` or `REFUSED`, and thought-attack responses as `HARMFUL`, `REFUSED`, or `HARMLESS`. Robustness is the fraction labeled `REFUSED` or `HARMLESS`; the undistracted ratio is the fraction of thought-attack responses labeled `HARMLESS`.

The judge saves its labels, scored sample logs, and `summary.json` with robustness, undistracted ratios, and 95% bootstrap intervals. The summary reports each prompt variant separately and averages all ten variants once each has 200 judged responses. Its scores replace the `r1_sorry` refusal filter's scores in the copied logs; the original logs are preserved. Rerun the same command to resume from saved judgments. Omit `--run` to inspect the prompts without making API calls. Invalid judgments are listed in `unresolved.jsonl`; retry them with `--retry-unresolved --run`, or supply reviewed labels with `--adjudications decisions.jsonl` (one JSON object per line with `id`, `label`, and `reason`).

OpenAI lists `o4-mini` for [retirement on October 23, 2026](https://developers.openai.com/api/docs/deprecations). To use another judge, pass `--judge-model MODEL`; report that choice alongside the results.

## Paper-to-code map

There are 30 reasoning configurations per benchmark and 60 jailbreak configurations. Configurations are kept as explicit YAML files so prompts and settings can be inspected directly.

| Paper result | Conditions / procedure |
| --- | --- |
| Figure 1, left; §4 | Five R1 models × six `identify_<thought>` conditions × five benchmarks. |
| Figure 1, right; §5 | Five R1 models × six `recover_<thought>_r1` conditions; compare against `baseline`. |
| Figure 2a–c | Short-irrelevant recovery across all 11 models: `recover_irrelevant_10p_{r1,s1,exaone}`. |
| Figure 2d | R1 irrelevant-thought lengths: 10%, 33%, 66%, and 100%. |
| Figure 3 | Six `aha_<thought>_r1` conditions, compared with uncued recovery and baseline. |
| Figure 4; §6 | Ten HarmBench prompts in both placements and all three model-family formats. |
| Figure 5; Appendix A.1 | `identify_correct` on five correct-thought subsets, compared with unhelpful identification. |
| Figure 6 / Table 2; Appendix A.2 | Answer-position yes/no probabilities for six identification conditions. Figure 6 averages dataset means; Table 2 pools rows and measures `P(no) < 0.05` or `P(no) >= 0.95`. See Appendix A.2 for the analysis. |
| Appendix A.3 | Manual categorization of failed Qwen-7B identification responses across six conditions. |
| Figure 7; Appendix A.4 | Qwen-7B/32B attention on 66 MATH-500 short-irrelevant examples, at least one model correct, at most 3,000 total tokens. See Appendix A.4 for the analysis. |
| Figure 8 | `baseline` on the full benchmark and the matched incorrect-thought subset. |
| Figure 9 | Six `instruct_<thought>_r1` conditions, compared with uncued recovery and baseline. |
| Tables 1, 3–10 | Dataset composition/construction, thought examples, and prompt/judge templates; see the dataset, task YAMLs, and paper. |

## Score the reasoning experiments

Each evaluation writes `results_*.json` and `samples_*.jsonl`. Keep the sample logs: the scoring script uses the generated responses and question identities to compute the final scores.

Run answer scoring after generation. For example, to score the MATH-500 recovery run above:

```bash
python scripts/score_reasoning.py \
  results/math500/recover_irrelevant_10p/qwen7b \
  --output results/scored/math500-recovery-qwen7b
```

You can also pass a directory containing runs across all five reasoning benchmarks. Use a separate directory for jailbreak runs. The script applies the notebook's answer extraction and normalization for math, multiple-choice, and yes/no answers. HumanEval uses the `pass@1` scores computed during evaluation.

The output contains `samples.csv` with per-question predictions and scores, and `summary.csv` with accuracy and 95% bootstrap intervals from 1,000 resamples. When multiple benchmarks are present, the `average` row gives each benchmark equal weight. Include all five benchmarks for the paper's five-dataset average. Choose a new output directory for each scoring run.

### Incorrect-thought comparisons

The incorrect-thought experiments use only questions for which R1-Distill Qwen-1.5B produced an incorrect answer during thought construction. Compare recovery with the baseline on those same questions. For example, MATH-500 uses 146 of its 500 questions for this comparison.

Pass the selected question indices as a JSON object keyed by benchmark, or derive them from the construction scores:

```bash
python scripts/score_reasoning.py results/reasoning \
  --incorrect-indices incorrect_indices.json \
  --output results/scored/reasoning
# Alternatively, replace --incorrect-indices with:
# --construction-input construction-scores.csv
```

The index file maps each benchmark to its zero-based dataset row indices, for example `{"math500": [0, 2]}` for a two-question selection. Construction scores need `model`, `task`, `index`, and `score` columns; `task` uses benchmark names such as `math500`. The script selects rows where the 1.5B model's score is zero, filters the incorrect-thought experiments, and adds `baseline_incorrect` results on the same questions. It also writes `incorrect_indices.json` for reuse.

The paper's selected indices are not included. The published `incorrect_thought` field also contains fallback thoughts for questions outside that subset, so a nonempty field does not identify an eligible question.

## Citation and license

```bibtex
@inproceedings{yang2025unhelpfulthoughts,
  title={How Well Can Reasoning Models Identify and Recover from Unhelpful Thoughts?},
  author={Sohee Yang and Sang-Woo Lee and Nora Kassner and Daniela Gottesman and Sebastian Riedel and Mor Geva},
  booktitle={Findings of EMNLP 2025},
  year={2025},
  url={https://aclanthology.org/2025.findings-emnlp.370}
}
```

Please cite the paper above and the evaluation harness using [`CITATION.bib`](CITATION.bib). The code is released under the [MIT license](LICENSE.md).
