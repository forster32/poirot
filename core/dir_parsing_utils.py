from hashlib import md5
import multiprocessing

import os
from dotenv import load_dotenv

from diskcache import Cache
from loguru import logger
from tqdm import tqdm

from config.client import PoirotConfig
from config.server import CACHE_DIRECTORY
from core.entities import Snippet
from utils.file_utils import read_file_with_fallback_encodings
from utils.tiktoken_utils import Tiktoken
from utils.code_validators import chunk_code
from utils.timer import Timer

chunk_cache = Cache(f'{CACHE_DIRECTORY}/chunk_cache') # we instantiate a singleton, diskcache will handle concurrency
file_name_cache = Cache(f'{CACHE_DIRECTORY}/file_name_cache')

tiktoken_client = Tiktoken()

def filter_file(directory: str, file: str, poirot_config: PoirotConfig) -> bool:
    cache_key = directory + file
    if cache_key in file_name_cache:
        return file_name_cache[cache_key]
    result = _filter_file(directory, file, poirot_config)
    file_name_cache[cache_key] = result
    return result

def _filter_file(directory: str, file: str, poirot_config: PoirotConfig) -> bool:
    """
    Check if a file should be filtered based on its size and other criteria.

    Args:
        file (str): The path to the file.
        poirot_config (PoirotConfig): The configuration object.

    Returns:
        bool: True if the file should be included, False otherwise.
    """
    # exclude files based on extension
    for ext in poirot_config.exclude_exts:
        if file.endswith(ext):
            return False
    only_file_name = file[len(directory) + 1 :]
    only_file_name_parts = only_file_name.split(os.path.sep)
    for dir_name in poirot_config.exclude_dirs:
        for file_part in only_file_name_parts[:-1]:
            if file_part == dir_name:
                return False
    for dir_name in poirot_config.exclude_path_dirs:
        if dir_name in only_file_name_parts:
            return False
    # exclude files based on size
    try:
        size = os.stat(file).st_size
        if size > 240000 or size < 10:
            return False
    except FileNotFoundError as e:
        logger.info(f"File not found: {file}. {e}")
        return False

    if not os.path.isfile(file):
        return False

    try:
        data = read_file_with_fallback_encodings(file)
    except UnicodeDecodeError:
        logger.warning(f"UnicodeDecodeError: {file}, skipping")
        return False
    if b'\x00' in data.encode():
        return False
    line_count = data.count("\n") + 1
    # if average line length is greater than 200, then it is likely not human readable
    if len(data) / line_count > 200:
        return False
    # check token density, if it is greater than 2, then it is likely not human readable
    token_count = tiktoken_client.count(data[:1000])
    if token_count == 0:
        return False
    if len(data[:1000]) / token_count < 2 and len(data) > 100:
        return False
    return True

def read_file(file_name: str) -> str:
    try:
        with open(file_name, "r") as f:
            return f.read()
    except Exception:
        return ""


FILE_THRESHOLD = 240

def conditional_hash(contents: str):
    if len(contents) > 255:
        return md5(contents.encode()).hexdigest()
    return contents

def file_path_to_chunks(file_path: str) -> list[str]:
    file_contents = read_file(file_path)
    content_hash = conditional_hash(file_path + file_contents)
    if content_hash in chunk_cache:
        return chunk_cache[content_hash]
    chunks = chunk_code(file_contents, path=file_path)
    if chunks:
        chunk_cache[content_hash] = chunks
    return chunks


# @file_cache()
def directory_to_chunks(
    directory: str, poirot_config: PoirotConfig,
) -> tuple[list[Snippet], list[str]]:

    logger.info(f"Reading files from {directory}")
    chunked_files = set()
    def traverse_dir(file_path: str = directory):
        only_file_name = os.path.basename(file_path)
        if only_file_name in ("node_modules", ".venv", "build", "venv", "patch"):
            return
        if file_path in chunked_files:
            return
        chunked_files.add(file_path)
        try:
            with os.scandir(file_path) as it:
                children = list(it)
                if len(children) > FILE_THRESHOLD:
                    return
                for entry in children:
                    if entry.is_dir(follow_symlinks=False):
                        yield from traverse_dir(entry.path)
                    else:
                        yield entry.path
        except NotADirectoryError:
            yield file_path
    with Timer():
        file_list = traverse_dir()
        file_list = [
            file_name
            for file_name in tqdm(file_list)
            if filter_file(directory, file_name, poirot_config)
        ]
    logger.info("Done reading files")
    all_chunks = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 4) as pool:
        for chunks in tqdm(pool.imap(file_path_to_chunks, file_list), total=len(file_list), desc="Chunking files"):
            all_chunks.extend(chunks)
    return all_chunks, file_list
