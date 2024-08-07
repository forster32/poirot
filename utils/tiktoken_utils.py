from __future__ import annotations


import tiktoken
from config.server import DEFAULT_GPT4_MODEL

TIKTOKEN_CACHE_DIR = "/tmp/cache/tiktoken"


class Tiktoken:
    def __init__(self):
        openai_models = [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-1106",
            "gpt-4",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-4-1106-preview",
            "gpt-4-0125-preview",
            "gpt-4-turbo-2024-04-09",
        ]
        self.openai_models = {
            model: tiktoken.encoding_for_model(model) for model in openai_models
        }

    def count(self, text: str, model: str = DEFAULT_GPT4_MODEL) -> int:
        return len(self.openai_models[model].encode(text, disallowed_special=()))

    def truncate_string(
        self, text: str, model: str = DEFAULT_GPT4_MODEL, max_tokens: int = 8192
    ) -> str:
        tokens = self.openai_models[model].encode(text)[:max_tokens - 1]
        return self.openai_models[model].decode(tokens)
