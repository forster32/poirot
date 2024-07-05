import os

from loguru import logger
from openai import AzureOpenAI, OpenAI

from config.server import (
    AZURE_API_KEY,
    DEFAULT_GPT4_MODEL,
    OPENAI_API_KEY,
    OPENAI_EMBEDDINGS_API_TYPE,
    OPENAI_EMBEDDINGS_AZURE_API_VERSION,
    OPENAI_EMBEDDINGS_AZURE_DEPLOYMENT,
    OPENAI_EMBEDDINGS_AZURE_ENDPOINT,
)
from core.entities import Message
# from logn.cache import file_cache
from utils.timer import Timer

SEED = 100

def get_embeddings_client() -> OpenAI | AzureOpenAI:
    client = None
    if OPENAI_EMBEDDINGS_API_TYPE == "openai":
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=90) if OPENAI_API_KEY else None
    elif OPENAI_EMBEDDINGS_API_TYPE == "azure":
        client = AzureOpenAI(
            azure_endpoint=OPENAI_EMBEDDINGS_AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            azure_deployment=OPENAI_EMBEDDINGS_AZURE_DEPLOYMENT,
            api_version=OPENAI_EMBEDDINGS_AZURE_API_VERSION,
        )
    if not client:
        raise ValueError("No Valid API key found for OpenAI or Azure!")
    return client
