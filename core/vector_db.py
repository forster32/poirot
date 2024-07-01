
import multiprocessing

import backoff
from diskcache import Cache
import numpy as np
import openai
import requests
from loguru import logger
from scipy.spatial.distance import cdist


from tqdm import tqdm
from utils.timer import Timer
from config.server import BATCH_SIZE, CACHE_DIRECTORY
from utils.hash import hash_sha256


from utils.openai_proxy import get_embeddings_client
from utils.tiktoken_utils import Tiktoken


CACHE_VERSION = "v0.0.1" 
tiktoken_client = Tiktoken()
vector_cache = Cache(f'{CACHE_DIRECTORY}/vector_cache') # we instantiate a singleton, diskcache will handle concurrency


@backoff.on_exception(
    backoff.expo,
    requests.exceptions.Timeout,
    max_tries=5,
)
def openai_with_expo_backoff(batch: tuple[str]):
    # check cache first
    embeddings: list[np.ndarray | None] = [None] * len(batch)
    cache_keys = [hash_sha256(text) + CACHE_VERSION for text in batch]

    try:
        for i, cache_key in enumerate(cache_keys):
            cache_value = vector_cache.get(cache_key)
            if cache_value is not None:
                embeddings[i] = cache_value
    except Exception as e:
        logger.warning(f"Error reading embeddings from cache: {e}")

    # not stored in cache, call openai
    batch = [
        text for i, text in enumerate(batch) if embeddings[i] is None
    ]  # remove all the cached values from the batch
    if len(batch) == 0:
        embeddings = np.array(embeddings)
        return embeddings  # all embeddings are in cache
    try:
        # make sure all token counts are within model params (max: 8192)
        new_embeddings = openai_call_embedding(batch)
    except requests.exceptions.Timeout as e:
        logger.exception(f"Timeout error occured while embedding: {e}")
    except Exception as e:
        logger.exception(e)
        if any(tiktoken_client.count(text) > 8192 for text in batch):
            logger.warning(
                f"Token count exceeded for batch: {max([tiktoken_client.count(text) for text in batch])} truncating down to 8192 tokens."
            )
            batch = [tiktoken_client.truncate_string(text) for text in batch]
            new_embeddings = openai_call_embedding(batch)
        else:
            raise e
    # get all indices where embeddings are None
    indices = [i for i, emb in enumerate(embeddings) if emb is None]
    # store the new embeddings in the correct position
    assert len(indices) == len(new_embeddings)
    for i, index in enumerate(indices):
        embeddings[index] = new_embeddings[i]
    # store in cache
    try:
        for cache_key, embedding in zip(cache_keys, embeddings):
            vector_cache.set(cache_key, embedding)
        embeddings = np.array(embeddings)
    except Exception as e:
        logger.warning(f"Error storing embeddings in cache: {e}")
    return embeddings


def embed_text_array(texts: list[str]) -> list[np.ndarray]:
    embeddings = []
    texts = [text if text else " " for text in texts]
    batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    workers = min(max(1, multiprocessing.cpu_count() // 4), 1)
    with Timer() as timer:
        if workers > 1 and len(batches) > 1:
            with multiprocessing.Pool(
                processes=workers
            ) as pool:
                embeddings = list(
                    tqdm(
                        pool.imap(openai_with_expo_backoff, batches),
                        total=len(batches),
                        desc="openai embedding",
                    )
                )
        else:
            embeddings = [openai_with_expo_backoff(batch) for batch in tqdm(batches, desc="openai embedding")]
    logger.info(f"Embedding docs took {timer.time_elapsed:.2f} seconds")
    return embeddings


def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)

# @redis_cache()
def openai_call_embedding_router(batch: list[str]):
    if len(batch) == 0:
        return np.array([])
    
    client = get_embeddings_client()
    response = client.embeddings.create(
        input=batch, model="text-embedding-3-small", encoding_format="float"
    )
    cut_dim = np.array([data.embedding for data in response.data])[:, :512]
    normalized_dim = normalize_l2(cut_dim)
    # save results to redis
    return normalized_dim



def openai_call_embedding(batch: list[str]):
    # Backoff on batch size by splitting the batch in half.
    try:
        return openai_call_embedding_router(batch)
    except openai.BadRequestError as e:
        # In the future we can better handle this by averaging the embeddings of the split batch
        if "maximum context length" in str(e):
            logger.warning(f"Token count exceeded for batch: {max([tiktoken_client.count(text) for text in batch])} truncating down to 8192 tokens.")
            batch = [tiktoken_client.truncate_string(text) for text in batch]
            return openai_call_embedding(batch)

def cosine_similarity(a, B):
    # use scipy
    return 1 - cdist(a, B, metric='cosine')

# @file_cache(ignore_params=["texts"])
def multi_get_query_texts_similarity(queries: list[str], documents: list[str]) -> list[float]:
    if not documents:
        return []
    embeddings = embed_text_array(documents)
    embeddings = np.concatenate(embeddings)
    with Timer() as timer:
        query_embedding = np.array(openai_call_embedding(queries))
    logger.info(f"Embedding query took {timer.time_elapsed:.2f} seconds")
    with Timer() as timer:
        similarity = cosine_similarity(query_embedding, embeddings)
    logger.info(f"Similarity took {timer.time_elapsed:.2f} seconds")
    similarity = similarity.tolist()
    return similarity
