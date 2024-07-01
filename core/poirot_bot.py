
import re

from core.chat import ChatGPT
from core.entities import (
    Message,
    Snippet,
)

from core.prompts import (
    context_files_to_change_prompt,
    context_files_to_change_system_prompt,
)

from core.planning_prompts import (
    issue_sub_request_system_prompt,
)


SNIPPET_TOKEN_BUDGET = int(150_000 * 3.5)  # 140k tokens
MAX_SNIPPETS = 15


def partition_snippets_if_test(snippets: list[Snippet], include_tests=False):
    if include_tests:
        return [snippet for snippet in snippets if "test" in snippet.file_path]
    return [snippet for snippet in snippets if "test" not in snippet.file_path]

def sort_and_fuse_snippets(
    snippets: list[Snippet],
    fuse_distance: int = 600,
) -> list[Snippet]:
    if len(snippets) <= 1:
        return snippets
    new_snippets = []
    snippets.sort(key=lambda x: x.start)
    current_snippet = snippets[0]
    for snippet in snippets[1:]:
        if current_snippet.end + fuse_distance >= snippet.start:
            current_snippet.end = max(current_snippet.end, snippet.end)
            current_snippet.score = max(current_snippet.score, snippet.score)
        else:
            new_snippets.append(current_snippet)
            current_snippet = snippet
    new_snippets.append(current_snippet)
    return new_snippets

def organize_snippets(snippets: list[Snippet], fuse_distance: int=600) -> list[Snippet]:
    """
    Fuse and dedup snippets that are contiguous. Combine ones of same file.
    """
    fused_snippets = []
    added_file_paths = set()
    for i, snippet in enumerate(snippets):
        if snippet.file_path in added_file_paths:
            continue
        added_file_paths.add(snippet.file_path)
        current_snippets = [snippet]
        for current_snippet in snippets[i + 1:]:
            if snippet.file_path == current_snippet.file_path:
                current_snippets.append(current_snippet)
        current_snippets = sort_and_fuse_snippets(current_snippets, fuse_distance=fuse_distance)
        fused_snippets.extend(current_snippets)
    return fused_snippets


def get_max_snippets(
    snippets: list[Snippet],
    budget: int = SNIPPET_TOKEN_BUDGET,
    expand: int = 300,
):
    """
    Start with max number of snippets and then remove then until the budget is met.
    Return the resulting organized snippets.
    """
    if not snippets:
        return []
    START_INDEX = min(len(snippets), MAX_SNIPPETS)
    for i in range(START_INDEX, 0, -1):
        expanded_snippets = [snippet.expand(expand * 2) if snippet.type_name == "source" else snippet for snippet in snippets[:i]]
        proposed_snippets = organize_snippets(expanded_snippets[:i])
        cost = sum([len(snippet.get_snippet(False, False)) for snippet in proposed_snippets])
        if cost <= budget:
            return proposed_snippets
    raise Exception("Budget number of chars too low!")

def parse_filenames(text):
    file_names = []
    possible_files = text.split("\n")
    # Regular expression pattern to match file names
    pattern = r'^[^\/.]+(\/[^\/.]+)*\.[^\/.]+$'
    for possible_file in possible_files:
        file_name = possible_file.strip()
        if re.match(pattern, file_name):
            file_names.append(file_name)
    # Find all occurrences of file names in the text
    return file_names


def context_get_files_to_change(
    relevant_snippets: list[Snippet],
    read_only_snippets: list[Snippet],
    problem_statement,
    # pr_diffs: str = "",
):
    messages: list[Message] = []
    messages.append(
        Message(role="system", content=issue_sub_request_system_prompt, key="system")
    )

    interleaved_snippets = []
    for i in range(max(len(relevant_snippets), len(read_only_snippets))):
        if i < len(relevant_snippets):
            interleaved_snippets.append(relevant_snippets[i])
        if i < len(read_only_snippets):
            interleaved_snippets.append(read_only_snippets[i])

    interleaved_snippets = partition_snippets_if_test(interleaved_snippets, include_tests=False)
    max_snippets = get_max_snippets(interleaved_snippets)
    if True:
        max_snippets = max_snippets[::-1]
    relevant_snippets = [snippet for snippet in max_snippets if any(snippet.file_path == relevant_snippet.file_path for relevant_snippet in relevant_snippets)]
    read_only_snippets = [snippet for snippet in max_snippets if not any(snippet.file_path == relevant_snippet.file_path for relevant_snippet in relevant_snippets)]

    relevant_snippet_template = '<relevant_file index="{i}">\n<file_path>\n{file_path}\n</file_path>\n<source>\n{content}\n</source>\n</relevant_file>'
    read_only_snippet_template = '<read_only_snippet index="{i}">\n<file_path>\n{file_path}\n</file_path>\n<source>\n{content}\n</source>\n</read_only_snippet>'
    # attach all relevant snippets
    joined_relevant_snippets = "\n".join(
        relevant_snippet_template.format(
            i=i,
            file_path=snippet.file_path,
            content=snippet.expand(300).get_snippet(add_lines=False) if snippet.type_name == "source" else snippet.get_snippet(add_lines=False),
        ) for i, snippet in enumerate(relevant_snippets)
    )
    relevant_snippets_message = f"# Relevant codebase files:\nHere are the relevant files from the codebase. These will be your primary reference to solve the problem:\n\n<relevant_files>\n{joined_relevant_snippets}\n</relevant_files>"
    messages.append(
        Message(
            role="user",
            content=relevant_snippets_message,
            key="relevant_snippets",
        )
    )
    joined_relevant_read_only_snippets = "\n".join(
        read_only_snippet_template.format(
            i=i,
            file_path=snippet.file_path,
            content=snippet.get_snippet(add_lines=False),
        ) for i, snippet in enumerate(read_only_snippets)
    )
    read_only_snippets_message = f"<relevant_read_only_snippets>\n{joined_relevant_read_only_snippets}\n</relevant_read_only_snippets>"
    if read_only_snippets:
        messages.append(
            Message(
                role="user",
                content=read_only_snippets_message,
                key="relevant_snippets",
            )
        )
    messages.append(
        Message(
            role="user",
            content=f"# GitHub Issue\n<issue>\n{problem_statement}\n</issue>",
        )
    )

    print("messages")
    for message in messages:
        print(message.content + "\n\n")
    joint_message = "\n\n".join(message.content for message in messages[1:])
    print("messages", joint_message)

    chat_gpt = ChatGPT(
        messages=[
            Message(
                role="system",
                content=context_files_to_change_system_prompt,
            ),
        ],
    )
    MODEL = "claude-3-opus-20240229"
    open("msg.txt", "w").write(joint_message + "\n\n" + context_files_to_change_prompt)
    files_to_change_response = chat_gpt.chat_anthropic(
        content=joint_message + "\n\n" + context_files_to_change_prompt,
        model=MODEL,
        temperature=0.1,
        # images=images,
    )
    relevant_files = []
    read_only_files = []
    # parse out <relevant_files> block
    relevant_files_pattern = re.compile(r"<relevant_files>(.*?)</relevant_files>", re.DOTALL)
    relevant_files_matches = relevant_files_pattern.findall(files_to_change_response)
    if relevant_files_matches:
        relevant_files_str = '\n'.join(relevant_files_matches)
        relevant_files = parse_filenames(relevant_files_str)
    # parse out <read_only_files> block
    read_only_files_pattern = re.compile(r"<read_only_files>(.*?)</read_only_files>", re.DOTALL)
    read_only_files_matches = read_only_files_pattern.findall(files_to_change_response)
    if read_only_files_matches:
        read_only_files_str = '\n'.join(read_only_files_matches)
        read_only_files = parse_filenames(read_only_files_str)
    relevant_files = list(dict.fromkeys(relevant_files))
    read_only_files = list(dict.fromkeys(read_only_files))
    return relevant_files, read_only_files
