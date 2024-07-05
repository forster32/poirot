import multiprocessing
import os
import re
import time
from collections.abc import Iterable

import tantivy
from diskcache import Cache
from loguru import logger
from tqdm import tqdm

from config.client import PoirotConfig
from config.server import CACHE_DIRECTORY
from core.dir_parsing_utils import directory_to_chunks
from core.entities import Snippet
from core.vector_db import multi_get_query_texts_similarity
from dataclass.files import Document
from utils.timer import Timer
from utils.streamable_functions import streamable


token_cache = Cache(f'{CACHE_DIRECTORY}/token_cache')
lexical_index_cache = Cache(f'{CACHE_DIRECTORY}/lexical_index_cache')
snippets_cache = Cache(f'{CACHE_DIRECTORY}/snippets_cache')
CACHE_VERSION = "v0.0.2"


schema_builder = tantivy.SchemaBuilder()
schema_builder.add_text_field("title", stored=True)
schema_builder.add_text_field("body", stored=True)
schema_builder.add_integer_field("doc_id", stored=True)
schema = schema_builder.build()


variable_pattern = re.compile(r"([A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$))")

def tokenize_code(code: str) -> str:
    matches = re.finditer(r"\b\w{2,}\b", code)
    tokens = []
    for m in matches:
        text = m.group()

        for section in text.split("_"):
            for part in variable_pattern.findall(section):
                if len(part) < 2:
                    continue
                # if len(part) < 5:
                #     tokens.append(part.lower())
                # if more than half of the characters are letters 
                # and the ratio of unique characters to the number of characters is less than 5
                if sum(1 for c in part if 'a' <= c <= 'z' or 'A' <= c <= 'Z' or '0' <= c <= '9') > len(part) // 2 \
                    and len(part) / len(set(part)) < 4:
                    tokens.append(part.lower())
    
    return " ".join(tokens)


class CustomIndex:
    def __init__(self, cache_path: str = None):
        os.makedirs(cache_path, exist_ok=True)
        self.index = tantivy.Index(schema) # pylint: disable=no-member
    
    def add_documents(self, documents: Iterable):
        writer = self.index.writer()
        for doc_id, (title, text) in enumerate(documents):
            writer.add_document(
                tantivy.Document( # pylint: disable=no-member
                    title=title,
                    body=text,
                    doc_id=doc_id
                )
            )
        writer.commit()
    
    def search_index(self, query: str) -> list[tuple[str, float, dict]]:
        query = tokenize_code(query)
        query = self.index.parse_query(query)
        searcher = self.index.searcher() # for some reason, the first searcher is empty
        for i in range(100):
            searcher = self.index.searcher()
            if searcher.num_docs > 0:
                break
            print(f"Index is empty, sleeping for {0.01 * i} seconds")
            time.sleep(0.01)
        else:
            raise Exception("Index is empty")
        results = searcher.search(query, limit=200).hits
        return [(searcher.doc(doc_id)["title"][0], score, searcher.doc(doc_id)) for score, doc_id in results]


def snippets_to_docs(snippets: list[Snippet], len_repo_cache_dir):
    docs = []
    for snippet in snippets:
        docs.append(
            Document(
                title=f"{snippet.file_path[len_repo_cache_dir:]}:{snippet.start}-{snippet.end}",
                content=snippet.get_snippet(add_ellipsis=False, add_lines=False),
            )
        )
    return docs

@streamable
def prepare_index_from_snippets(
    snippets: list[Snippet],
    len_repo_cache_dir: int = 0,
    cache_path: str = None,
) -> CustomIndex | None:
    all_docs: list[Document] = snippets_to_docs(snippets, len_repo_cache_dir)
    if len(all_docs) == 0:
        return None
    index = CustomIndex(
        cache_path=cache_path
    )
    yield "Tokenizing documents...", index
    all_tokens = []
    try:
        with Timer() as timer:
            for doc in all_docs:
                all_tokens.append(token_cache.get(doc.content + CACHE_VERSION))
            misses = [i for i, token in enumerate(all_tokens) if token is None]
            workers = multiprocessing.cpu_count() // 2
            if workers > 1:
                with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as p:
                    missed_tokens = p.map(
                        tokenize_code,
                        tqdm(
                            [all_docs[i].content for i in misses],
                            total=len(misses),
                            desc="Tokenizing documents"
                        )
                    )
            else:
                missed_tokens = [
                    tokenize_code(all_docs[i].content) for i in misses
                ]
            for i, token in enumerate(missed_tokens):
                all_tokens[misses[i]] = token
                token_cache[all_docs[misses[i]].content + CACHE_VERSION] = token
        logger.debug(f"Tokenizing documents took {timer.time_elapsed} seconds")
        yield "Building lexical index...", index
        all_titles = [doc.title for doc in all_docs]
        with Timer() as timer:
            index.add_documents(
                tqdm(zip(all_titles, all_tokens), total=len(all_docs), desc="Indexing")
            )
        logger.debug(f"Indexing took {timer.time_elapsed} seconds")
    except FileNotFoundError as e:
        logger.exception(e)

    yield "Index built", index
    return index


def search_index(query: str, index: CustomIndex):
    """Search the index based on a query.

    This function takes a query and an index as input and returns a dictionary of document IDs
    and their corresponding scores.
    """
    # Create a query parser for the "content" field of the index
    results_with_metadata = index.search_index(query)
    # Search the index
    res = {}
    for doc_id, score, _ in results_with_metadata:
        if doc_id not in res:
            res[doc_id] = score
    # min max normalize scores from 0.5 to 1
    if len(res) == 0:
        max_score = 1
        min_score = 0
    else:
        max_score = max(res.values())
        min_score = min(res.values()) if min(res.values()) < max_score else 0
    res = {k: (v - min_score) / (max_score - min_score) for k, v in res.items()}
    return res


SNIPPET_FORMAT = """File path: {file_path}

{contents}"""

# @file_cache(ignore_params=["snippets"])
def compute_vector_search_scores(queries: list[str], snippets: list[Snippet]):
    # get get dict of snippet to score
    with Timer() as timer:
        snippet_str_to_contents = {
            snippet.denotation: SNIPPET_FORMAT.format(
                file_path=snippet.file_path,
                contents=snippet.get_snippet(add_ellipsis=False, add_lines=False),
            )
            for snippet in snippets
        }
    logger.info(f"Snippet to contents took {timer.time_elapsed:.2f} seconds")
    snippet_contents_array = list(snippet_str_to_contents.values())
    multi_query_snippet_similarities = multi_get_query_texts_similarity(
        queries, snippet_contents_array
    )
    snippet_denotations = [snippet.denotation for snippet in snippets]
    snippet_denotation_to_scores = [{
        snippet_denotations[i]: score
        for i, score in enumerate(query_snippet_similarities)
    } for query_snippet_similarities in multi_query_snippet_similarities]
    return snippet_denotation_to_scores


def get_lexical_cache_key(
    root_directory: str, 
    seed: str = ""
):
    root_directory = os.path.basename(root_directory)
    return f"{root_directory}_{CACHE_VERSION}_{seed}"


@streamable
def prepare_lexical_search_index(
    root_directory: str,
    poirot_config: PoirotConfig,
    seed: str = "" # used for lexical cache key
):
    lexical_cache_key = get_lexical_cache_key(root_directory, seed=seed)

    yield "Collecting snippets...", [], None
    snippets_results = snippets_cache.get(lexical_cache_key)
    if snippets_results is None:
        snippets, file_list = directory_to_chunks(
            root_directory, poirot_config
        )
        snippets_cache[lexical_cache_key] = snippets, file_list
    else:
        snippets, file_list = snippets_results

    yield "Building index...", snippets, None

    index = prepare_index_from_snippets(
        snippets,
        len_repo_cache_dir=len(root_directory) + 1,
        cache_path=f"{CACHE_DIRECTORY}/lexical_index_cache/{lexical_cache_key}"
    )
    
    yield "Lexical index built.", snippets, index

    return snippets, index

if __name__ == "__main__":
    root_directory = os.getenv("ROOT_DIRECTORY")
    sweep_config = PoirotConfig()
    assert root_directory
    import time
    start = time.time()
    snippets , index = prepare_lexical_search_index(root_directory, sweep_config, start)
    print(snippets, index)
    result = search_index("logger export", index)
    print("Time taken:", time.time() - start)
    # print some of the keys
    print(list(result.keys())[:5])
    # print the first 2 result keys sorting by value
    print(sorted(result.items(), key=lambda x: result.get(x, 0), reverse=True)[:5])
