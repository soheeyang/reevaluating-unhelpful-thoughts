from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter

import re


@register_filter("custom")
class CustomFilter(Filter):
    """
    Custom filter that applies a custom, user-defined function to the model responses.
    """

    def __init__(self, **kwargs) -> None:
        self.filter_fn = kwargs.pop("filter_fn")

        super().__init__(**kwargs)

    def apply(self, resps, docs):
        return self.filter_fn(resps, docs)


@register_filter("r1_mcqa_last")
class McqaFilterLast(Filter):
    """A filter that extracts values from text using regex pattern matching.

    This filter applies a regex pattern to each model response and extracts the value
    that is matched at the last.
    If no match is found, returns a fallback value.
    """

    def __init__(
        self,
        fallback: str = "[invalid]",
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.fallback = fallback

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        
        def process_response(text):
            # Split by double newlines and get the last paragraph
            text = text.strip()
            if not text:
                return self.fallback
            
            # Try different patterns to extract the answer
            patterns = [
                r"\\boxed\{\s*([A-Za-z]+)\s*\}",          # \boxed{A}
                r"The best answer is ([A-Za-z]+)[\.\s]",   # "The best answer is A." or "The best answer is A " 
                r"The answer is ([A-Za-z]+)[\.\s]",        # "The answer is A." or "The answer is A "
                r"(?i)Answer:\s*([A-Za-z]+)",             # "Answer: A" (case insensitive)
                r"\*\*Answer:\*\*\s*([A-Za-z]+)",         # "**Answer:** A"
                r"\[([A-Z]+)\]",                       # [A]
                r"\*\*([A-Za-z]+)\*\*",                   # **A**
                r"\(([A-Za-z]+)\)",                       # (A)
                r"([A-Za-z]+)\)",                         # A)
                r"^([A-Za-z]+)$",                          # Just "A" on its own
                r"(?i)best answer is\s+([A-Za-z]+)",      # "best answer is A" with anything after
                r"(?i)answer is\s+([A-Za-z]+)",           # "answer is A" with anything after
            ]
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, text))
                if matches:
                    return matches[-1].group(1)
            
            # If still no patterns match
            return self.fallback
        
        return [[process_response(r) for r in resp] for resp in resps]


@register_filter("r1_mcqa")
class McqaFilter(Filter):
    """A filter that extracts values from text using regex pattern matching.

    This filter applies a regex pattern to each model response and extracts the value
    in the last paragraph that is matched at the first.
    If no match is found, returns a fallback value.
    """

    def __init__(
        self,
        fallback: str = "[invalid]",
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.fallback = fallback

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        
        def process_response(text):
            # Split by double newlines and get the last paragraph
            paragraphs = text.split('\n\n')
            if not paragraphs:
                return self.fallback
            
            last_paragraph = paragraphs[-1].strip()
            if not last_paragraph:
                return self.fallback
            
            # Try different patterns to extract the answer
            patterns = [
                r"The best answer is ([A-Z])[\.\s]",   # "The best answer is A." or "The best answer is A " 
                r"The answer is ([A-Z])[\.\s]",        # "The answer is A." or "The answer is A "
                r"(?i)Answer:\s*([A-Z])",             # "Answer: A" (case insensitive)
                r"\*\*Answer:\*\*\s*([A-Z])",         # "**Answer:** A"
                r"\[([A-Z])\]",                       # [A]
                r"\\boxed\{\s*([A-Z])\s*\}",          # \boxed{A}
                r"\*\*([A-Z])\*\*",                   # **A**
                r"\(([A-Z])\)",                       # (A)
                r"([A-Z])\)",                         # A)
                r"^([A-Z])$"                          # Just "A" on its own
            ]
            
            for pattern in patterns:
                match = re.search(pattern, last_paragraph)
                if match:
                    return match.group(1)
            
            # If no patterns match, try one more strategy - find any uppercase letter
            # after "best answer is" or "answer is"
            final_patterns = [
                r"(?i)best answer is\s+([A-Z])",      # "best answer is A" with anything after
                r"(?i)answer is\s+([A-Z])"           # "answer is A" with anything after
            ]
            
            for pattern in final_patterns:
                match = re.search(pattern, last_paragraph)
                if match:
                    return match.group(1)
            
            # If still no patterns match
            return self.fallback
        
        return [[process_response(r) for r in resp] for resp in resps]
    

@register_filter("r1_boxed")
class BoxedFilter(Filter):
    """A filter that extracts values from text using regex pattern matching.

    This filter applies a regex pattern to each model response and extracts matched values
    that are inside \boxed{}.
    """

    def __init__(
        self,
        fallback: str = "[invalid]",
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.fallback = fallback

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:

        def process_response(text):
            paragraphs = text.split('\n\n')
            if not paragraphs:
                return self.fallback
            
            last_paragraph = paragraphs[-1].strip()
            if not last_paragraph:
                return self.fallback
            
            # Find the \boxed command
            match = re.search(r"\\boxed\{", last_paragraph)
            if not match:
                return self.fallback
                
            # Find the position where \boxed{ starts
            start_pos = match.end()
            
            # Stack to keep track of opening and closing braces
            stack = 1  # Already encountered the opening brace from \boxed{
            
            for i in range(start_pos, len(last_paragraph)):
                if last_paragraph[i] == '{':
                    stack += 1
                elif last_paragraph[i] == '}':
                    stack -= 1
                
                # When the stack is empty, we've found the matching closing brace
                if stack == 0:
                    # Extract the content between \boxed{ and the matching }
                    return last_paragraph[start_pos:i].strip()
            
            # If we never find the matching closing brace
            return self.fallback
        
        return [[process_response(r) for r in resp] for resp in resps]


# WARNING: This should not be used for the actual evaluation because it is highly inaccurate
# LLM-based evaluation should be used instead.
@register_filter("r1_sorry")
class SorryFilter(Filter):
    """A filter that extracts values from text using regex pattern matching.

    This filter applies a regex pattern to each model response and extracts matched values.
    If no match is found, returns a fallback value. Useful for extracting structured data
    (like numbers) from unstructured model outputs.
    """

    def __init__(
        self,
        fallback: str = "[invalid]",
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.fallback = fallback

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:

        def process_response(text):

            # Pattern for the first format: <｜Assistant｜><think>content</think>\nanswer
            pattern1 = r'</think>(.*)'

            # Try the first pattern
            match = re.search(pattern1, text, re.DOTALL)
            if match:
                text = match.group(1).strip()

            if 'sorry' in text.lower():
                return "refused"
            
            elif 'boxed' in text.lower():
                return "refused"

            return self.fallback
        
        return [[process_response(r) for r in resp] for resp in resps]