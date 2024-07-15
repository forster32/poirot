import os
from dotenv import load_dotenv
load_dotenv()

load_dotenv(dotenv_path=".env", override=True, verbose=True)

DEFAULT_GPT4_MODEL = os.environ.get("DEFAULT_GPT4_MODEL", "gpt-4-turbo-2024-04-09")

BATCH_SIZE = int(
    os.environ.get("BATCH_SIZE", 256) # Voyage only allows 128 items per batch and 120000 tokens per batch
)

ROOT_DIRECTORY = os.environ.get("ROOT_DIRECTORY", None)
CACHE_DIRECTORY = os.environ.get("CACHE_DIRECTORY", ".agent_caches")

MQ_HOST = os.environ.get("MQ_HOST", "localhost")
MQ_PORT = int(os.environ.get("MQ_PORT", 5672))
MQ_USER = os.environ.get("MQ_USER", "guest")
MQ_PASSWORD = os.environ.get("MQ_PASSWORD", "guest")
MQ_QUEUE = os.environ.get("MQ_QUEUE", "default_queue")
