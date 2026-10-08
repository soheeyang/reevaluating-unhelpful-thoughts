import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def r1_parse_code(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    import re
    pattern = r"```python\s*(.*?)\s*```"
    
    def extract_code(text):
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:  # If matches were found
            return matches[-1]  # Return the last match
        else:
            return "[invalid]"  # Return fallback if no matches
    
    return [[extract_code(r) for r in resp] for resp in resps]