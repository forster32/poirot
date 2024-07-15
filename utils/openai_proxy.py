import os

from loguru import logger
from openai import OpenAI

from core.entities import Message
# from logn.cache import file_cache
from utils.timer import Timer

SEED = 100

def get_embeddings_client() -> OpenAI:
    client = OpenAI(timeout=90)
    if not client:
        raise ValueError("No Valid API key found for OpenAI!")
    return client
