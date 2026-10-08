"""Answer extraction and normalization for the reasoning experiments.

HumanEval pass@1 is read from the evaluation log; no generated code runs here.
Math answers use the paper's normalization rules and exact string comparison.
"""

import ast
import operator
import re
import string
from difflib import SequenceMatcher

import numpy as np
import sympy as sp


FALLBACK = "[invalid]"


def extract_response(text):
    """
    Extract the answer text that comes after the thought tags.

    Args:
        text (str): The text containing both thought and answer components.

    Returns:
        str: The answer text after the thought tags, or empty string if not found.
    """
    # Pattern for the first format: <｜Assistant｜><think>content</think>\nanswer
    pattern1 = r'</think>(.*)'

    # Pattern for the second format: <|im_start|>assistant\n<|im_start|>think\ncontent\n<|im_start|>answer\nanswer
    pattern2 = r'<\|im_start\|>answer(.*)'

    pattern3 = r'</thought>(.*)'

    # Try the first pattern
    match = re.search(pattern1, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If first pattern doesn't match, try the second pattern
    match = re.search(pattern2, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If first pattern doesn't match, try the second pattern
    match = re.search(pattern3, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Return empty string if neither pattern matches
    return text


def McqaFilter(text, prompt, similarity_match=False):
    text = text.strip()
    if not text:
        return FALLBACK

    # First try direct patterns to extract a letter from the answer text
    direct_patterns = [
        r"\\boxed\{\s*([A-E])\s*\}",          # \boxed{A}
        r"\boxed\{\s*([A-E])\s*\}",           # \boxed{A}
        r"boxed\{\s*([A-E])\s*\}",           # \boxed{A}
        r"(?i)Answer:\s*([A-E])",             # "Answer: A" (case insensitive)
        r"(?i)ANSWER:\s*([A-E])",             # "Answer: A" (case insensitive)
        r"\*\*Answer:\*\*\s*([A-E])",         # "**Answer:** A"
        r"\*\*Answer\*\*:\s*([A-E])",         # "**Answer**: A" (colon outside asterisks)
        r"\*\*Answer\*\*\s*:\s*([A-E])",      # "**Answer** : A" (flexible colon position)
        r"\*\*([A-E])\*\*",                   # **A**
        r"\[([A-E])\]",                       # [A]
        r"\(([A-E])\)",                       # (A)
        r"([A-E])\)",                         # A)
        r"^([A-E])$",                         # Just "A" on its own
        r"The best answer is ([A-E])[\.\s]",  # "The best answer is A." or "The best answer is A "
        r"The answer is ([A-E])[\.\s]",       # "The answer is A." or "The answer is A "
        r"(?i)best answer is\s+([A-E])",      # "best answer is A" with anything after
        r"(?i)answer is\s+([A-E])",           # "answer is A" with anything after
        r"\*\*([A-E])[\.|\)]\s",              # **C. or **C) followed by a space
        r"\*\*([A-E])[\.\)]\s+[^*]+\*\*",     # **C. text** or **C) text**
        r"\*\*([A-E])[\.\)].+?\*\*",          # **C. anything** or **C) anything**
        r"\*\*([A-E])\*\*.+",                 # **C** followed by anything
    ]

    for pattern in direct_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]

    # Parse the choices from the prompt - handling the special format
    choices = {}
    # Use a more robust pattern to extract choices from the special format
    choice_pattern = r'\(([A-E])\)\s*(\d+(?:\s*\d+)*)'
    choice_matches = re.findall(choice_pattern, prompt)

    for letter, choice_text in choice_matches:
        choices[letter] = choice_text.strip()

    # Extract the boxed content from the answer text - handling both escaped and non-escaped
    boxed_patterns = [
        r'\\boxed\{(.*?)(?:\}|$)',  # For escaped backslash
        r'\boxed\{(.*?)(?:\}|$)',     # For literal backslash
        r'boxed\{(.*?)(?:\}|$)'     # For literal backslash
    ]

    boxed_content = None
    for pattern in boxed_patterns:
        boxed_match = re.search(pattern, text)
        if boxed_match:
            boxed_content = boxed_match.group(1).strip()
            break

    if boxed_content and choices:
        # Clean the boxed content
        clean_boxed = re.sub(r'\\text\{|\}', '', boxed_content).strip()

        # If it's a direct letter, return it
        if re.match(r'^[A-E]$', clean_boxed):
            return clean_boxed

        if similarity_match:

            best_match = None
            highest_similarity = -1

            for letter, choice_text in choices.items():
                # First check for exact numeric match (most reliable)
                boxed_num = re.match(r'^(\d+)', clean_boxed)
                choice_num = re.match(r'^(\d+)', choice_text)

                if boxed_num and choice_num and boxed_num.group(1) == choice_num.group(1):
                    # Add a high bonus to prioritize numeric matches
                    similarity = 100 + len(boxed_num.group(1))
                else:
                    # Use sequence matcher for text similarity
                    similarity = SequenceMatcher(None, choice_text.lower(), clean_boxed.lower()).ratio()

                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = letter

            if best_match:
                return best_match

    if similarity_match:

        # Last chance - look for a direct number in the boxed content that matches a choice
        if boxed_content:
            clean_boxed = re.sub(r'\\text\{|\}', '', boxed_content).strip()
            boxed_num = re.match(r'^(\d+)', clean_boxed)

            if boxed_num:
                for letter, choice_text in choices.items():
                    if choice_text.strip() == boxed_num.group(1):
                        return letter

    return FALLBACK


def extract_answer_number(response):
    # Pattern to find a number after "Answer:" or similar response
    pattern = r'(?:answer:|\*\*answer:\*\*)\s*(\d+(?:\.\d+)?)'

    # Case insensitive search
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1)

    # If no match found with "Answer:", try to find any number
    alt_pattern = r'\b(\d+(?:\.\d+)?)\b'
    matches = re.findall(alt_pattern, response)
    if matches:
        return matches[0]  # Return the first number found

    return FALLBACK


def BoxedFilter(response, prompt):
    response = response.strip()
    if not response:
        return FALLBACK

    matches = list(re.finditer(r"boxed\{", response))
    if not matches:
        return extract_answer_number(response)

    match = matches[-1]

    # Find the position where \boxed{ starts
    start_pos = match.end()

    # Stack to keep track of opening and closing braces
    stack = 1  # Already encountered the opening brace from \boxed{

    for i in range(start_pos, len(response)):
        if response[i] == '{':
            stack += 1
        elif response[i] == '}':
            stack -= 1

        # When the stack is empty, we've found the matching closing brace
        if stack == 0:
            # Extract the content between \boxed{ and the matching }
            return response[start_pos:i].strip()

    # If we never find the matching closing brace
    return FALLBACK


def HelpfulBoxedFilter(response, prompt):
    response = response.strip()
    if not response:
        return FALLBACK

    if response.strip() == "no":
        return "no"

    if response.strip() == "yes":
        return "yes"

    matches = list(re.finditer(r"boxed\{", response))
    if not matches:
        return extract_answer_number(response)

    match = matches[-1]

    # Find the position where \boxed{ starts
    start_pos = match.end()

    # Stack to keep track of opening and closing braces
    stack = 1  # Already encountered the opening brace from \boxed{

    for i in range(start_pos, len(response)):
        if response[i] == '{':
            stack += 1
        elif response[i] == '}':
            stack -= 1

        # When the stack is empty, we've found the matching closing brace
        if stack == 0:
            # Extract the content between \boxed{ and the matching }
            return response[start_pos:i].strip()

    # If we never find the matching closing brace
    return FALLBACK


def extract_letter_from_parentheses(input_string):
    # Pattern to match a single alphabet character inside parentheses
    pattern = r"\(([A-Za-z])\)"

    match = re.search(pattern, input_string)
    if match:
        return match.group(1)  # Return just the letter
    else:
        return input_string


def reformat_negative_fractions(input_string):
    # Pattern to match \frac{numerator}{denominator} with possible negative signs
    pattern = r'\\frac\{(-?\d+)\}\{(-?\d+)\}'

    def replace_match(match):
        numerator = match.group(1)
        denominator = match.group(2)

        # Remove negative signs for processing
        num_negative = numerator.startswith('-')
        denom_negative = denominator.startswith('-')

        # Clean numerator and denominator (remove negative signs)
        clean_num = numerator[1:] if num_negative else numerator
        clean_denom = denominator[1:] if denom_negative else denominator

        # Determine the sign of the fraction
        if (num_negative and not denom_negative) or (not num_negative and denom_negative):
            # One negative sign = result is negative
            return f"-\\frac{{{clean_num}}}{{{clean_denom}}}"
        elif num_negative and denom_negative:
            # Two negative signs = result is positive
            return f"\\frac{{{clean_num}}}{{{clean_denom}}}"
        else:
            # No negative signs = result is positive
            return f"\\frac{{{numerator}}}{{{denominator}}}"

    # Replace all occurrences
    result = re.sub(pattern, replace_match, input_string)
    return result


def remove_text_command(input_string):
    # Pattern to match \text{...} with possible nested braces
    pattern = r"\\text\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"

    # Function to process each match
    def replace_match(match):
        # Return just the content inside \text{...}
        return match.group(1)

    # Replace all occurrences
    result = re.sub(pattern, replace_match, input_string)
    return result


def format_twodigit_fractions(input_string):
    # Pattern to match \frac followed by one or two digits with optional space
    pattern = r'\\frac\s*(\d)(?:\s*)(\d)'

    # Function to process each match
    def replace_match(match):
        numerator = match.group(1)
        denominator = match.group(2)
        return f"\\frac{{{numerator}}}{{{denominator}}}"

    # Replace all occurrences
    result = re.sub(pattern, replace_match, input_string)
    return result


def extract_complex_number(text):
    # This pattern makes the LaTeX-style delimiters \( and \) optional
    # It extracts the complex number expression after an equals sign
    pattern = r'\\?\(?\s*[a-zA-Z]\s*=\s*([-+]?\d*\.?\d*\s*[-+]\s*\d*\.?\d*i)\s*\\?\)?'

    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()

    # For cases with just real or just imaginary part
    alt_pattern = r'\\?\(?\s*[a-zA-Z]\s*=\s*([-+]?\d*\.?\d*i|[-+]?\d*\.?\d*)\s*\\?\)?'
    alt_match = re.search(alt_pattern, text)
    if alt_match:
        return alt_match.group(1).strip()

    return text


def normalize_decimal(text):
    """
    Ensures decimal numbers have a leading zero when starting with a decimal point

    Args:
        text (str): Text that may contain decimal numbers

    Returns:
        str: Text with properly formatted decimal numbers
    """
    # Replace decimals that start with a point with a leading 0
    return re.sub(r'(^|[^0-9])\.([0-9])', r'\g<1>0.\g<2>', text)


def normalize_fraction(latex_expr):
    """
    Normalizes LaTeX fractions to the standard form \frac{numerator}{denominator}

    Args:
        latex_expr (str): LaTeX fraction expression which might be in various forms

    Returns:
        str: Normalized LaTeX fraction
    """
    # Pattern for \frac{num}{den}
    standard_pattern = r'\\frac\{(.*?)\}\{(.*?)\}'

    # Pattern for \frac{num}den where den doesn't have braces
    unbraced_den_pattern = r'\\frac\{(.*?)\}([^{].*?)(?:\s|$)'

    # Pattern for \fracnum{den} where num doesn't have braces
    unbraced_num_pattern = r'\\frac([^{].*?)\{(.*?)\}'

    # Check if it's already in standard form
    if re.match(standard_pattern, latex_expr):
        return latex_expr

    # Check for unbraced denominator
    match_den = re.match(unbraced_den_pattern, latex_expr)
    if match_den:
        numerator = match_den.group(1)
        denominator = match_den.group(2)
        return f"\\frac{{{numerator}}}{{{denominator}}}"

    # Check for unbraced numerator
    match_num = re.match(unbraced_num_pattern, latex_expr)
    if match_num:
        numerator = match_num.group(1)
        denominator = match_num.group(2)
        return f"\\frac{{{numerator}}}{{{denominator}}}"

    return latex_expr


# The notebook used sympy.sympify on model-written strings. Reconstruct the
# allowed mathematical operations from a Python AST instead: no eval, attribute
# lookup, subscripts, arbitrary function calls, imports, or string arguments.
_SAFE_MATH_FUNCTIONS = {
    name: getattr(sp, name) for name in (
        "sqrt", "cbrt", "root", "Abs", "abs", "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "exp", "log", "ln", "floor", "ceiling", "factorial",
        "Rational", "Integer", "Float",
    ) if hasattr(sp, name)
}
_SAFE_MATH_FUNCTIONS["abs"] = sp.Abs
_SAFE_MATH_CONSTANTS = {"pi": sp.pi, "E": sp.E, "I": sp.I, "oo": sp.oo, "nan": sp.nan}
_SAFE_BINARY_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def _safe_math_expression(text):
    """Parse mathematical expressions without executing generated Python.

    Unsupported syntax raises ValueError, which evaluate_fraction handles by
    returning the unchanged normalized fraction. This is the one intentional
    security-related deviation from the notebook's normalization implementation.
    """
    if len(text) > 4096:
        raise ValueError("Math expression is too long")
    # sympify treats ^ as exponentiation rather than Python's integer XOR.
    tree = ast.parse(text.replace("^", "**"), mode="eval")

    def convert(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return sp.Integer(node.value) if type(node.value) is int else sp.Float(node.value)
        if isinstance(node, ast.Name):
            return _SAFE_MATH_CONSTANTS.get(node.id, sp.Symbol(node.id))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = convert(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINARY_OPERATORS:
            return _SAFE_BINARY_OPERATORS[type(node.op)](convert(node.left), convert(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = _SAFE_MATH_FUNCTIONS.get(node.func.id)
            if function is not None and not node.keywords:
                return function(*(convert(argument) for argument in node.args))
        raise ValueError("Unsupported math-expression syntax")

    return convert(tree.body)


def evaluate_fraction(latex_expr):
    """
    Evaluates a LaTeX fraction if it contains only numbers,
    otherwise leaves it in symbolic form.

    Args:
        latex_expr (str): LaTeX fraction expression in various formats

    Returns:
        Union[float, str]: Evaluated fraction or normalized expression
    """
    # First normalize the fraction to standard form
    normalized_expr = normalize_fraction(latex_expr)

    # Extract numerator and denominator from the normalized expression
    pattern = r'\\frac\{(.*?)\}\{(.*?)\}'
    match = re.match(pattern, normalized_expr)

    if not match:
        return normalized_expr  # Return normalized version if not a fraction

    numerator = match.group(1)
    denominator = match.group(2)

    # Try to convert to numeric values
    try:
        num = _safe_math_expression(numerator)
        den = _safe_math_expression(denominator)

        # Check if both are numbers (no symbols)
        if num.is_number and den.is_number:
            return str(float(num) / float(den))
        else:
            # Contains symbols, return normalized expression
            return normalized_expr
    except:
        # If any error occurs during parsing, return the normalized expression
        return normalized_expr


def convert_simple_fractions_to_latex(input_string):
    # Pattern to match simple fractions like -1/3, 2/3, 5/3, etc.
    pattern = r'(-?\d+)/(\d+)'

    # Function to process each match
    def replace_match(match):
        numerator = match.group(1)
        denominator = match.group(2)
        return f"\\frac{{{numerator}}}{{{denominator}}}"

    # Replace all occurrences
    result = re.sub(pattern, replace_match, input_string)
    return result


def simplify_subscripts(input_string):
    # Pattern to match _{single digit or character}
    pattern = r'_\{([0-9a-zA-Z])\}'

    # Function to process each match
    def replace_match(match):
        subscript = match.group(1)
        return f"_{subscript}"

    # Replace all occurrences
    result = re.sub(pattern, replace_match, input_string)
    return result


def remove_unnecessary_decimals(number_string):
    # First check if there's an underscore
    if '_' in number_string:
        return number_string  # Preserve strings with underscores

    # Try to convert the string to a float
    try:
        num = float(number_string)

        # Check if the number is equivalent to its integer value
        if num == int(num):
            return str(int(num))
        else:
            return number_string
    except ValueError:
        # If conversion fails, return the original string
        return number_string


def math500_filter(text):
    text = remove_text_command(text)

    for from_, to_ in [
        ('dfrac', 'frac'),
        ('^\\circ', ''),
        ('\\left(', '('),
        ('\\right)', ')'),
        ('degrees', ''),
        (r',\!', ''),
        (r'\$', ''),
        (r'\mbox{ inches}^2', ''),
        (r'\mbox{ cm}^2', ''),
        (r'\mbox{ feet}^2', ''),
        ('cents', ''),
        ('inches per second', ''),
        ('inches', ''),
        ('meters', ''),
        ('^{mathrm{th}} grade', ''),
        ('%', ''),
        ('feet', ''),
        ('minutes', ''),
        (', cm', ''),
        ('^{mathrm{th}}\ngrade', ''),
        ('^{th}', ''),
    ]:
        text = text.replace(from_, to_)

    pattern = r'\\sqrt\{(\d+\.?\d*)\}'
    text = re.sub(pattern, r'\\sqrt\1', text)

    text = extract_complex_number(text)
    text = normalize_decimal(text)

    text = convert_simple_fractions_to_latex(text)

    text = format_twodigit_fractions(text)

    text = normalize_fraction(text)
    text = reformat_negative_fractions(text)
    text = evaluate_fraction(text)

    text = text.replace('\\', '')

    text = extract_letter_from_parentheses(text)

    text = simplify_subscripts(text)

    text = remove_unnecessary_decimals(text)

    return text


def remove_leading_zeros(number_string):
    # Check if the string has digits only
    if number_string.isdigit():
        # Convert to int and back to string to remove leading zeros
        return str(int(number_string))
    else:
        # If it's not just digits (contains other characters), return as is
        return number_string


def aime24_filter(text):
    text = remove_leading_zeros(text)

    for from_, to_ in [
        ('dfrac', 'frac'),
        ('^\\circ', ''),
        ('\\left(', '('),
        ('\\right)', ')'),
        ('degrees', ''),
        (r',\!', ''),
        (r'\$', ''),
        (r'\mbox{ inches}^2', ''),
        (r'\mbox{ cm}^2', ''),
        (r'\mbox{ feet}^2', ''),
        ('cents', ''),
        ('inches per second', ''),
        ('inches', ''),
        ('meters', ''),
        ('^{mathrm{th}} grade', ''),
        ('%', ''),
        ('feet', ''),
        ('minutes', ''),
        (', cm', ''),
        ('^{mathrm{th}}\ngrade', ''),
        ('^{th}', ''),
    ]:
        text = text.replace(from_, to_)

    return text


def exact_match(
    predictions,
    references,
    regexes_to_ignore=None,
    ignore_case=False,
    ignore_punctuation=False,
    ignore_numbers=False,
):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            predictions = np.array([[re.sub(s, "", x) for x in prediction] for prediction in predictions])
            references = np.array([re.sub(s, "", x) for x in references])
    else:
        predictions = np.asarray(predictions)
        references = np.asarray(references)

    if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    score_list = predictions == references

    return {"exact_match": np.mean(score_list)}


# Exact-match options used for the paper scores.
_MATH_METRIC = {
    "ignore_case": True,
    "ignore_punctuation": False,
    "regexes_to_ignore": [",", r"\$", "(?s).*#### ", r"\.$"],
}
_MCQA_METRIC = {"ignore_case": True, "ignore_punctuation": True}
config_dict = {
    "aime24": {"metric_list": [_MATH_METRIC]},
    "arc": {"metric_list": [_MCQA_METRIC]},
    "gpqa": {"metric_list": [_MCQA_METRIC]},
    "humaneval": {"metric_list": [{}]},
    "math500": {"metric_list": [_MATH_METRIC]},
}
config_dict.update({"identify_" + task: config for task, config in list(config_dict.items())})

response_filter_dict = {
    "aime24": BoxedFilter,
    "arc": McqaFilter,
    "gpqa": McqaFilter,
    "humaneval": lambda text, prompt: text,
    "math500": BoxedFilter,
}
response_filter_dict.update({"identify_" + task: HelpfulBoxedFilter for task in list(response_filter_dict)})
em_filter_dict = {
    "aime24": aime24_filter,
    "arc": lambda text: text,
    "gpqa": lambda text: text,
    "humaneval": lambda text: text,
    "math500": math500_filter,
}
em_filter_dict.update({"identify_" + task: lambda text: text for task in list(em_filter_dict)})


def get_prediction(row, **kwargs):
    task = row['task']
    response = row['response']
    prompt = row['prompt']
    response_filter = response_filter_dict[task]
    em_filter = em_filter_dict[task]

    result = em_filter(response_filter(response, prompt, **kwargs))
    return result


def get_gold(row):
    task = row['task']
    gold = row['target']
    em_filter = em_filter_dict[task]
    gold = em_filter(gold)

    return gold


def get_score(row):
    task = row['task']

    if ('humaneval' in task) and ('identify' not in task):
        return row['score']

    metric_config = config_dict[task]['metric_list'][0]

    regexes_to_ignore = metric_config.get('regexes_to_ignore', None)
    ignore_case = metric_config.get('ignore_case', False)
    ignore_punctuation = metric_config.get('ignore_punctuation', False)
    ignore_numbers = metric_config.get('ignore_numbers', False)

    result = row['prediction']
    gold = row['gold']

    score = exact_match(
        [[result]],
        [gold],
        regexes_to_ignore,
        ignore_case,
        ignore_punctuation,
        ignore_numbers,
    )['exact_match']
    return score

