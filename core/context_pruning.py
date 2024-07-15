import urllib
from dataclasses import dataclass, field

from loguru import logger

from core.entities import Snippet
from utils.repository_utils import Repository

ASSISTANT_MAX_CHARS = 4096 * 4 * 0.95  # ~95% of 4k tokens
NUM_SNIPPETS_TO_SHOW_AT_START = 15
MAX_REFLECTIONS = 1
MAX_ITERATIONS = 25
NUM_ROLLOUTS = 1 # dev speed
SCORE_THRESHOLD = 8 # good score
STOP_AFTER_SCORE_THRESHOLD_IDX = 0 # stop after the first good score and past this index
MAX_PARALLEL_FUNCTION_CALLS = 1
NUM_BAD_FUNCTION_CALLS = 5

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

    @property
    def top_snippet_paths(self):
        return [snippet.file_path for snippet in self.current_top_snippets]

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

# TODO
# add import trees for any relevant_file_paths (code files that appear in query)
def build_import_trees(
    rcm: RepoContextManager,
    # import_graph: nx.DiGraph,
    # override_import_graph: nx.DiGraph = None,
) -> tuple[RepoContextManager]:
    # if import_graph is None and override_import_graph is None:
    #     return rcm
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
