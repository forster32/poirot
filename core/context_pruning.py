from copy import deepcopy
from math import log
import subprocess
import urllib
from dataclasses import dataclass, field

# import networkx as nx
# import openai
from loguru import logger
# from openai.types.beta.thread import Thread
# from openai.types.beta.threads.run import Run

from config.client import PoirotConfig
from core.chat import ChatGPT
from core.entities import Message, Snippet
# from logn.cache import file_cache
# from utils.chat_logger import ChatLogger
# from utils.convert_openai_anthropic import AnthropicFunctionCall, mock_function_calls_to_string
from utils.repository_utils import Repository
# from utils.ripgrep_utils import post_process_rg_output
# from utils.openai_listwise_reranker import listwise_rerank_snippets
# from utils.progress import AssistantConversation, TicketProgress
from utils.tree_utils import DirectoryTree

ASSISTANT_MAX_CHARS = 4096 * 4 * 0.95  # ~95% of 4k tokens
NUM_SNIPPETS_TO_SHOW_AT_START = 15
MAX_REFLECTIONS = 1
MAX_ITERATIONS = 25
NUM_ROLLOUTS = 1 # dev speed
SCORE_THRESHOLD = 8 # good score
STOP_AFTER_SCORE_THRESHOLD_IDX = 0 # stop after the first good score and past this index
MAX_PARALLEL_FUNCTION_CALLS = 1
NUM_BAD_FUNCTION_CALLS = 5


anthropic_function_calls = """<tool_description>
<tool_name>code_search</tool_name>
<description>
Passes the code_entity into ripgrep to search the entire codebase and return a list of files and line numbers where it appears. Useful for finding definitions, usages, and references to types, classes, functions, and other entities that may be relevant. Review the search results using `view_files` to determine relevance and discover new files to explore.
</description>
<parameters>
<parameter>
<name>analysis</name>
<type>string</type>
<description>Explain what new information you expect to discover from this search and why it's needed to get to the root of the issue. Focus on unknowns rather than already stored information.</description>
</parameter>
<parameter>
<name>code_entity</name>
<type>string</type>
<description>
The code entity to search for. This must be a distinctive name, not a generic term. For functions, search for the definition syntax, e.g. 'def foo' in Python or 'function bar' or 'const bar' in JavaScript. Trace dependencies of critical functions/classes, follow imports to find definitions, and explore how key entities are used across the codebase.
</description>
</parameter>
</parameters>
</tool_description>

<tool_description>
<tool_name>view_files</tool_name>
<description>
Retrieves the contents of the specified file(s). After viewing new files, use `code_search` on relevant entities to continue discovering potentially relevant files. You may view three files per tool call. Prioritize viewing new files over ones that are already stored.
</description>
<parameters>
<parameter>
<name>analysis</name>
<type>string</type>
<description>Explain what new information viewing these files will provide and why it's necessary to resolve the issue. Avoid restating already known information.</description>
</parameter>
<parameter>
<name>first_file_path</name>
<type>string</type>
<description>The path of a new file to view.</description>
</parameter>
<parameter>
<name>second_file_path</name>
<type>string</type>
<description>The path of another new file to view (optional).</description>
</parameter>
<parameter>
<name>third_file_path</name>
<type>string</type>
<description>The path of a third new file to view (optional).</description>
</parameter>
</parameters>
</tool_description>

<tool_description>
<tool_name>store_file</tool_name>
<description>
Adds a newly discovered file that provides important context or may need modifications to the list of stored files. You may only store one new file per tool call. Avoid storing files that have already been added.
</description>
<parameters>
<parameter>
<name>analysis</name>
<type>string</type>
<description>Explain what new information this file provides, why it's important for understanding and resolving the issue, and what potentially needs to be modified. Include a brief supporting code excerpt.</description>
</parameter>
<parameter>
<name>file_path</name>
<type>string</type>
<description>The path of the newly discovered relevant file to store.</description>
</parameter>
</parameters>
</tool_description>

You MUST call the tools using this exact XML format:

<function_call>
<invoke>
<tool_name>$TOOL_NAME</tool_name>
<parameters>
<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
...
</parameters>
</invoke>
</function_call>

Here is an example illustrating a complex code search to discover new relevant information:

<example>
<function_call>
<invoke>
<tool_name>code_search</tool_name>
<parameters>
<analysis>The get_user_by_id method likely queries from a User model or database table. I need to search for references to "User" to find where and how user records are defined, queried and filtered in order to determine what changes are needed to support excluding deleted users from the get_user_by_id results.</analysis>
<code_entity>User</code_entity>
</parameters>
</invoke>
</function_call>
</example>

Remember, your goal is to discover and store ALL files that are relevant to solving the issue. Perform targeted searches to uncover new information, view new files to understand the codebase, and avoid re-analyzing already stored files."""

sys_prompt = """You are a brilliant engineer assigned to solve the following GitHub issue. Your task is to search through the codebase and locate ALL files that are RELEVANT to resolving the issue. A file is considered RELEVANT if it provides important context or may need to be modified as part of the solution.

You will begin with a small set of stored relevant files. However, it is critical that you identify every additional relevant file by exhaustively searching the codebase. Your goal is to generate an extremely comprehensive list of files for an intern engineer who is completely unfamiliar with the codebase. Prioritize finding all relevant files over perfect precision - it's better to include a few extra files than to miss a key one.

To accomplish this, you will iteratively search for and view new files to gather all the necessary information. Follow these steps:

1. Perform targeted code searches to find definitions, usages, and references for ALL unknown variables, classes, attributes, functions and other entities that may be relevant based on the currently stored files and issue description. Be creative and think critically about what to search for to get to the root of the issue. 

2. View new files from the search results that seem relevant. Avoid viewing files that are already stored, and instead focus on discovering new information.

3. Store additional files that provide important context or may need changes based on the search results, viewed files, and issue description. 

Repeat steps 1-3, searching and exploring the codebase exhaustively until you are confident you have found all relevant files. Prioritize discovering new information over re-analyzing what is already known.

Here are the tools at your disposal:
""" + anthropic_function_calls

unformatted_user_prompt = """\
## Stored Files
DO NOT CALL THE STORE OR VIEW TOOLS ON THEM AGAIN AS THEY HAVE ALREADY BEEN STORED.
<stored_files>
{snippets_in_repo}
</stored_files>

{import_tree_prompt}
## User Request
<user_request>
{query}
<user_request>"""

PLAN_SUBMITTED_MESSAGE = "SUCCESS: Report and plan submitted."

@dataclass
class RepoContextManager:
    repository: Repository
    current_top_snippets: list[Snippet] = field(default_factory=list)
    read_only_snippets: list[Snippet] = field(default_factory=list)
    test_current_top_snippets: list[Snippet] = field(default_factory=list)
    issue_report_and_plan: str = ""
    import_trees: str = ""
    relevant_file_paths: list[str] = field(
        default_factory=list
    )  # a list of file paths that appear in the user query
    # UNUSED:
    snippets: list[Snippet] = field(default_factory=list) # This is actually used in benchmarks
    snippet_scores: dict[str, float] = field(default_factory=dict)
    current_top_tree: str | None = None
    dir_obj: DirectoryTree | None = None

    @property
    def top_snippet_paths(self):
        return [snippet.file_path for snippet in self.current_top_snippets]

    @property
    def relevant_read_only_snippet_paths(self):
        return [snippet.file_path for snippet in self.read_only_snippets]

    def format_context(
        self,
        unformatted_user_prompt: str,
        query: str,
    ):
        files_in_repo_str = ""
        stored_files = set()
        for idx, snippet in enumerate(list(dict.fromkeys(self.current_top_snippets))[:NUM_SNIPPETS_TO_SHOW_AT_START]):
            if snippet.file_path in stored_files:
                continue
            stored_files.add(snippet.file_path)
            snippet_str = \
f'''
<stored_file index="{idx + 1}">
<file_path>{snippet.file_path}</file_path>
<source>
{snippet.content}
</source>
</stored_file>
'''
            files_in_repo_str += snippet_str
        repo_tree = str(self.dir_obj)
        import_tree_prompt = """
## Import trees for code files in the user request
<import_trees>
{import_trees}
</import_trees>
"""
        import_tree_prompt = (
            import_tree_prompt.format(import_trees=self.import_trees.strip("\n"))
            if self.import_trees
            else ""
        )
        user_prompt = unformatted_user_prompt.format(
            query=query,
            snippets_in_repo=files_in_repo_str,
            repo_tree=repo_tree,
            import_tree_prompt=import_tree_prompt,
            file_paths_in_query=", ".join(self.relevant_file_paths),
        )
        return user_prompt

    def add_snippets(self, snippets_to_add: list[Snippet]):
        # self.dir_obj.add_file_paths([snippet.file_path for snippet in snippets_to_add])
        for snippet in snippets_to_add:
            self.current_top_snippets.append(snippet)

    def boost_snippets_to_top(self, snippets_to_boost: list[Snippet], code_files_in_query: list[str]):
        # self.dir_obj.add_file_paths([snippet.file_path for snippet in snippets_to_boost])
        for snippet in snippets_to_boost:
            # get first positions of all snippets that are in the code_files_in_query
            all_first_in_query_positions = [self.top_snippet_paths.index(file_path) for file_path in code_files_in_query if file_path in self.top_snippet_paths]
            last_mentioned_result_index = (max(all_first_in_query_positions, default=-1) + 1) if all_first_in_query_positions else 0
            # insert after the last mentioned result
            self.current_top_snippets.insert(max(0, last_mentioned_result_index), snippet)

    def add_import_trees(self, import_trees: str):
        self.import_trees += "\n" + import_trees

    def append_relevant_file_paths(self, relevant_file_paths: str):
        # do not use append, it modifies the list in place and will update it for ALL instances of RepoContextManager
        self.relevant_file_paths = self.relevant_file_paths + [relevant_file_paths]

    def set_relevant_paths(self, relevant_file_paths: list[str]):
        self.relevant_file_paths = relevant_file_paths

    def update_issue_report_and_plan(self, new_issue_report_and_plan: str):
        self.issue_report_and_plan = new_issue_report_and_plan

# add import trees for any relevant_file_paths (code files that appear in query)
def build_import_trees(
    rcm: RepoContextManager,
    import_graph: None,
    # import_graph: nx.DiGraph,
    # override_import_graph: nx.DiGraph = None,
) -> tuple[RepoContextManager]:
    # if import_graph is None and override_import_graph is None:
    #     return rcm
    if import_graph is None:
        return rcm
    # if override_import_graph:
    #     import_graph = override_import_graph
    # if we have found relevant_file_paths in the query, we build their import trees
    # code_files_in_query = rcm.relevant_file_paths
    # # graph_retrieved_files = graph_retrieval(rcm.top_snippet_paths, rcm, import_graph)[:15]
    # graph_retrieved_files = [snippet.file_path for snippet in rcm.read_only_snippets]
    # if code_files_in_query:
    #     for file in code_files_in_query:
    #         # fetch direct parent and children
    #         representation = (
    #             f"\nThe file '{file}' has the following import structure: \n"
    #             + build_full_hierarchy(import_graph, file, 2)
    #         )
    #         if graph_retrieved_files:
    #             representation += "\n\nThe following modules may contain helpful services or utility functions:\n- " + "\n- ".join(graph_retrieved_files)
    #         rcm.add_import_trees(representation)
    # # if there are no code_files_in_query, we build import trees for the top 5 snippets
    # else:
    #     for snippet in rcm.current_top_snippets[:5]:
    #         file_path = snippet.file_path
    #         representation = (
    #             f"\nThe file '{file_path}' has the following import structure: \n"
    #             + build_full_hierarchy(import_graph, file_path, 2)
    #         )
    #         if graph_retrieved_files:
    #             representation += "\n\nThe following modules may contain helpful services or utility functions:\n- " + "\n-".join(graph_retrieved_files)
    #         rcm.add_import_trees(representation)
    # return rcm


# add any code files that appear in the query to current_top_snippets
def add_relevant_files_to_top_snippets(rcm: RepoContextManager) -> RepoContextManager:
    code_files_in_query = rcm.relevant_file_paths
    for file in code_files_in_query:
        current_top_snippet_paths = [
            snippet.file_path for snippet in rcm.current_top_snippets
        ]
        # if our mentioned code file isnt already in the current_top_snippets we add it
        if file not in current_top_snippet_paths:
            try:
                code_snippets = [
                    snippet for snippet in rcm.snippets if snippet.file_path == file
                ]
                rcm.boost_snippets_to_top(code_snippets, code_files_in_query)
            except Exception as e:
                logger.error(
                    f"Tried to add code file found in query but recieved error: {e}, skipping and continuing to next one."
                )
    return rcm

def generate_import_graph_text(graph):
  # Create a dictionary to store the import relationships
  import_dict = {}

  # Iterate over each node (file) in the graph
  for node in graph.nodes():
    # Get the files imported by the current file
    imported_files = list(graph.successors(node))

    # Add the import relationships to the dictionary
    if imported_files:
      import_dict[node] = imported_files
    else:
      import_dict[node] = []

  # Generate the text-based representation
  final_text = ""
  visited_files = set()
  for file, imported_files in sorted(import_dict.items(), key=lambda x: x[0]):
    if file not in visited_files:
      final_text += generate_file_imports(graph, file, visited_files, "")
      final_text += "\n"

  # Add files that are not importing any other files
  non_importing_files = [
      file for file, imported_files in import_dict.items()
      if not imported_files and file not in visited_files
  ]
  if non_importing_files:
    final_text += "\n".join(non_importing_files)

  return final_text


def generate_file_imports(graph,
                          file,
                          visited_files,
                          last_successor,
                          indent_level=0):
  # if you just added this file as a successor, you don't need to add it again
  visited_files.add(file)
  text = "  " * indent_level + f"{file}\n" if file != last_successor else ""

  for imported_file in graph.successors(file):
    text += "  " * (indent_level + 1) + f"──> {imported_file}\n"
    if imported_file not in visited_files:
      text += generate_file_imports(graph, imported_file, visited_files,
                                    imported_file, indent_level + 2)

  return text

# fetch all files mentioned in the user query
def parse_query_for_files(
    query: str, rcm: RepoContextManager
) -> tuple[RepoContextManager, None]:
    MAX_FILES_TO_ADD = 5
    code_files_to_add = []
    code_files_to_check = set(list(rcm.repository.get_file_list()))
    code_files_uri_encoded = [
        urllib.parse.quote(file_path) for file_path in code_files_to_check
    ]
    # check if any code files are mentioned in the query
    for file, file_uri_encoded in zip(code_files_to_check, code_files_uri_encoded):
        if file in query or file_uri_encoded in query:
            code_files_to_add.append((file, file))
        # check this separately to match a/b/c.py with b/c.py
        elif len(file.split('/')) >= 2:
            last_two_parts = '/'.join(file.split('/')[-2:])
            if (last_two_parts in query or urllib.parse.quote(last_two_parts) in query) and len(last_two_parts) > 5:
                # this can be improved by bumping out other matches if it's a "better" match (e.g. more specific)
                code_files_to_add.append((file, last_two_parts))
    # sort by where the match was found in the query
    code_files_to_add = sorted(
        code_files_to_add,
        key=lambda x: query.index(x[1]) if x[1] in query else urllib.parse.unquote(query).index(x[1]), # must exist in query because we matched something
    )
    # convert to a deduplicated list of file paths
    code_files_to_add = list(dict.fromkeys([file for file, _ in code_files_to_add]))
    for code_file in code_files_to_add[:MAX_FILES_TO_ADD]:
        rcm.append_relevant_file_paths(code_file)
    return rcm, None


# do not ignore repo_context_manager
# @file_cache(ignore_params=["seed", "ticket_progress", "chat_logger"])
def get_relevant_context(
    query: str,
    repo_context_manager: RepoContextManager,
    seed: int = None,
    # import_graph: nx.DiGraph = None,
    num_rollouts: int = NUM_ROLLOUTS,
    ticket_progress = None,
    chat_logger = None,
) -> RepoContextManager:
    logger.info("Seed: " + str(seed))
    try:
        # for any code file mentioned in the query, build its import tree - This is currently not used
        repo_context_manager = build_import_trees(
            repo_context_manager,
            # import_graph,
        )
        # for any code file mentioned in the query add it to the top relevant snippets
        repo_context_manager = add_relevant_files_to_top_snippets(repo_context_manager)
        # add relevant files to dir_obj inside repo_context_manager, this is in case dir_obj is too large when as a string
        repo_context_manager.dir_obj.add_relevant_files(
            repo_context_manager.relevant_file_paths
        )

        user_prompt = repo_context_manager.format_context(
            unformatted_user_prompt=unformatted_user_prompt,
            query=query,
        )
        return repo_context_manager
    except Exception as e:
        logger.exception(e)
        return repo_context_manager

# if __name__ == "__main__":
#     try:
#         from utils.github_utils import get_installation_id
#         from utils.ticket_utils import prep_snippets

#         organization_name = "sweepai"
#         installation_id = get_installation_id(organization_name)
#         cloned_repo = ClonedRepo("sweepai/sweep", installation_id, "main")
#         query = "allow 'sweep.yaml' to be read from the user/organization's .github repository. this is found in client.py and we need to change this to optionally read from .github/sweep.yaml if it exists there"
#         # golden response is
#         # sweepai/handlers/create_pr.py:401-428
#         # sweepai/config/client.py:178-282
#         ticket_progress = TicketProgress(
#             tracking_id="test",
#         )
#         snippets = prep_snippets(cloned_repo, query, ticket_progress)
#         rcm = get_relevant_context(
#             query,
#             snippets, # THIS SHOULD BE BROKEN
#             ticket_progress,
#             chat_logger=ChatLogger({"username": "wwzeng1"}),
#         )
#         for snippet in rcm.current_top_snippets:
#             print(snippet.denotation)
#     except Exception as e:
#         logger.error(f"context_pruning.py failed to run successfully with error: {e}")
#         raise e
