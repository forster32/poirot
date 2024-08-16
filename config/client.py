import os

from loguru import logger
from pydantic import BaseModel


class PoirotConfig(BaseModel):
    include_dirs: list[str] = []
    exclude_dirs: list[str] = [
        ".git",
        "node_modules",
        "build",
        ".venv",
        "venv",
        "patch",
        "packages/blobs",
        "dist",
        "oa3gen",
    ]
    exclude_path_dirs: list[str] = [
        "node_modules", 
        "build", 
        ".venv", 
        "venv", 
        ".git", 
        "dist"]
    exclude_substrings_aggressive: list[str] = [ # aggressively filter out file paths, may drop some relevant files
        "integration",
        ".spec",
        ".test",
        ".json",
        "test"
    ]
    include_exts: list[str] = [
        ".cs",
        ".csharp",
        ".py",
        ".md",
        ".txt",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
    ]
    exclude_exts: list[str] = [
        ".min.js",
        ".min.js.map",
        ".min.css",
        ".min.css.map",
        ".tfstate",
        ".tfstate.backup",
        ".jar",
        ".ipynb",
        ".png",
        ".jpg",
        ".jpeg",
        ".download",
        ".gif",
        ".bmp",
        ".tiff",
        ".ico",
        ".mp3",
        ".wav",
        ".wma",
        ".ogg",
        ".flac",
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".patch",
        ".patch.disabled",
        ".wmv",
        ".m4a",
        ".m4v",
        ".3gp",
        ".3g2",
        ".rm",
        ".swf",
        ".flv",
        ".iso",
        ".bin",
        ".tar",
        ".zip",
        ".7z",
        ".gz",
        ".rar",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".svg",
        ".parquet",
        ".pyc",
        ".pub",
        ".pem",
        ".ttf",
        ".dfn",
        ".dfm",
        ".feature",
        "sweep.yaml",
        "pnpm-lock.yaml",
        "LICENSE",
        'package-lock.json',
        # 'package.json',
        'pyproject.toml',
        'requirements.txt',
        'yarn.lock',
        '.lockb',
        '.gitignore',
        '.lock'
    ]

    # returns if file is excluded or not
    def is_file_excluded(self, file_path: str) -> bool:
        parts = file_path.split(os.path.sep)
        for i, part in enumerate(parts):
            if part in self.exclude_dirs:
                return True
        # check extension of file
        file_name = parts[-1]
        for ext in self.exclude_exts:
            if file_name.endswith(ext):
                return True
        # if there is not extension, then it is likely bad
        # Because some documents cancel this judgment. "Dockerfile", "Makefile", "Rakefile", "Procfile", "Vagrantfile", "Gemfile"
        # if "." not in file_name:
        #     return True
        return False

