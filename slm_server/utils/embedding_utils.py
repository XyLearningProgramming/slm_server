from typing import Callable

from slm_server.model import EmbeddingInput

DetokenizeFunc = Callable[[list[int], list[int] | None, bool], bytes]


def process_embedding_input(
    input_data: EmbeddingInput, detokenize: DetokenizeFunc
) -> str | list[str]:
    """Process embedding input, converting tokens to text if needed."""
    if (
        input_data
        and isinstance(input_data, list)
        and not isinstance(input_data[0], str)
    ):
        # Check if it's a list of integers (single tokenized input)
        if isinstance(input_data[0], int):
            # Convert tokens back to text using the model's tokenizer
            return detokenize(input_data).decode("utf-8", errors="ignore")
        # Check if it's a list of list of integers (multiple tokenized inputs)
        elif (
            isinstance(input_data[0], list)
            and len(input_data[0]) > 0
            and isinstance(input_data[0][0], int)
        ):
            # Convert each tokenized input back to text
            return [
                detokenize(tokens).decode("utf-8", errors="ignore")
                for tokens in input_data
            ]

    return input_data
