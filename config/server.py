import os
from dotenv import load_dotenv
load_dotenv()

load_dotenv(dotenv_path=".env", override=True, verbose=True)

# DEFAULT_GPT4_MODEL = os.environ.get("DEFAULT_GPT4_MODEL", "gpt-4-0125-preview")
DEFAULT_GPT4_MODEL = os.environ.get("DEFAULT_GPT4_MODEL", "gpt-4-turbo-2024-04-09")



AZURE_API_KEY = os.environ.get("AZURE_API_KEY", None)
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", None)

MULTI_REGION_CONFIG = os.environ.get("MULTI_REGION_CONFIG", None)
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", None)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
OPENAI_API_TYPE = os.environ.get("OPENAI_API_TYPE", None)
OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION", None)
OPENAI_EMBEDDINGS_API_TYPE = os.environ.get("OPENAI_EMBEDDINGS_API_TYPE", None)
OPENAI_EMBEDDINGS_AZURE_API_VERSION = os.environ.get("OPENAI_EMBEDDINGS_AZURE_API_VERSION", None)
OPENAI_EMBEDDINGS_AZURE_DEPLOYMENT = os.environ.get("OPENAI_EMBEDDINGS_AZURE_DEPLOYMENT", None)
OPENAI_EMBEDDINGS_AZURE_ENDPOINT = os.environ.get("OPENAI_EMBEDDINGS_AZURE_ENDPOINT", None)

BATCH_SIZE = int(
    os.environ.get("BATCH_SIZE", 256) # Voyage only allows 128 items per batch and 120000 tokens per batch
)


CACHE_DIRECTORY = os.environ.get("CACHE_DIRECTORY", "caches")

