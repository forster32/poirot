from pydantic import BaseModel, Field
from typing import Literal

class Message(BaseModel):
    # role: (
    #     Literal["system"] | Literal["user"] | Literal["assistant"] | Literal["function"]
    # )
    role: Literal["system", "user", "assistant", "function"]
    content: str | None = None
    name: str | None = None
    function_call: dict | None = None
    key: str | None = None
    annotations: dict | None = None

    # @classmethod
    # def from_tuple(cls, tup: tuple[str | None, str | None]) -> Self:
    #     if tup[0] is None:
    #         return cls(role="assistant", content=tup[1])
    #     else:
    #         return cls(role="user", content=tup[0])

    def to_openai(self) -> str:
        obj = {
            "role": self.role,
            "content": self.content,
        }
        if self.function_call:
            obj["function_call"] = self.function_call
        if self.role == "function":
            obj["name"] = self.name
        return obj
    

class Snippet(BaseModel):
    content: str = Field(repr=False)
    start: int
    end: int
    file_path: str
    score: float = 0.0
    type_name: Literal["source", "tests", "dependencies", "tools", "docs"] = "source"

    def __eq__(self, other):
        if isinstance(other, Snippet):
            return (
                self.file_path == other.file_path
                and self.start == other.start
                and self.end == other.end
            )
        return False

    def __hash__(self):
        return hash((self.file_path, self.start, self.end))

    def get_snippet(self, add_ellipsis: bool = True, add_lines: bool = True):
        lines = self.content.splitlines()
        snippet = "\n".join(
            (f"{i + self.start}: {line}" if add_lines else line)
            for i, line in enumerate(lines[max(self.start - 1, 0) : self.end])
        )
        if add_ellipsis:
            if self.start > 1:
                snippet = "...\n" + snippet
            if self.end < self.content.count("\n") + 1:
                snippet = snippet + "\n..."
        return snippet

    def expand(self, num_lines: int = 25):
        return Snippet(
            content=self.content,
            start=max(self.start - num_lines, 1),
            end=min(self.end + num_lines, self.content.count("\n") + 1),
            file_path=self.file_path,
            score=self.score,
        )

    @property
    def denotation(self):
        return f"{self.file_path}:{self.start}-{self.end}"
