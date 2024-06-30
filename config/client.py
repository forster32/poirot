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
        'package.json',
        'pyproject.toml',
        'requirements.txt',
        'yarn.lock',
        '.lockb',
        '.gitignore',
        '.lock'
    ]
    excluded_languages: list[str] = [
        'TOML',
        'Git Attributes',
        'XML Property List',
    ]
    # cutoff for when we output truncated versions of strings, this is an arbitrary number and can be changed
    truncation_cutoff: int = 20000
    # Image formats
    max_file_limit: int = 60_000
    # github comments
    max_github_comment_body_length: int = 65535
    # allowed image types for vision
    allowed_image_types: list[str] = [
        "jpg",
        "jpeg",
        "webp",
        "png"
    ]

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.dict())

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "PoirotConfig":
        data = yaml.safe_load(yaml_str)
        return cls.parse_obj(data)

    
    # returns if file is excluded or not
    def is_file_excluded(self, file_path: str) -> bool:
        parts = file_path.split(os.path.sep)
        for i, part in enumerate(parts):
            if part in self.exclude_dirs:
                return True
            # check extension of file
            if i == len(parts) - 1:
                for ext in self.exclude_exts:
                    if part.endswith(ext):
                        return True
                # if there is not extension, then it is likely bad
                if "." not in part:
                    return True
        return False
    
    # returns if file is excluded or not, this version may drop actual relevant files
    def is_file_excluded_aggressive(self, dir: str, file_path: str) -> bool:
        # tiktoken_client = Tiktoken()
        # must exist
        if not os.path.exists(os.path.join(dir, file_path)) and not os.path.exists(file_path):
            return True
        full_path = os.path.join(dir, file_path)
        if os.stat(full_path).st_size > 240000 or os.stat(full_path).st_size < 5:
            return True
        # exclude binary 
        with open(full_path, "rb") as f:
            is_binary = False
            for block in iter(lambda: f.read(1024), b""):
                if b"\0" in block:
                    is_binary = True
                    break
            if is_binary:
                return True
        try:
            # fetch file
            data = read_file_with_fallback_encodings(full_path)
            lines = data.split("\n")
        except UnicodeDecodeError:
            logger.warning(f"UnicodeDecodeError in is_file_excluded_aggressive: {full_path}, skipping")
            return True
        line_count = len(lines)
        # if average line length is greater than 200, then it is likely not human readable
        if len(data)/line_count > 200:
            return True
    
        # check token density, if it is greater than 2, then it is likely not human readable
        # token_count = tiktoken_client.count(data)
        # if token_count == 0:
        #     return True
        # if len(data)/token_count < 2:
        #     return True
        
        # now check the file name
        parts = file_path.split(os.path.sep)
        for part in parts:
            if part in self.exclude_dirs or part in self.exclude_exts:
                return True
        for part in self.exclude_substrings_aggressive:
            if part in file_path:
                return True
            
        # check if file is autogenerated
        auto_generated, _ = self.is_file_auto_generated(file_path)
        if auto_generated:
            return True
        return False
    
    # checks the actual context of a file to see if it is suitable for sweep or not
    # for example checks for size and composition of the file_contents
    # returns False if the file is bad
    def is_file_suitable(self, file_contents: str) -> tuple[bool, str]:
        if file_contents is None:
            return False, "The file contents were a None Type object, this is most likely an issue on our end."
        try:
            encoded_file = encode_file_with_fallback_encodings(file_contents)
        except UnicodeEncodeError as e:
            logger.warning(f"Failed to encode file: {e}")
            return False, "Failed to encode file!"
        # file is too large or too small
        file_length = len(encoded_file)
        if file_length > 240000:
            return False, "The size of this file means it is likely auto generated."
        lines = file_contents.split("\n")
        line_count = len(lines)
        # if average line length is greater than 200, then it is likely not human readable
        if line_count == 0:
            return False, "Line count for this file was 0!"
        if len(file_contents)/line_count > 200:
            return False, "This file was determined to be non human readable due to the average line length."
        return True, ""
    
    def is_file_bad(self, file_name: str, repo_dir: str) -> tuple[bool, str]:
        """
        Uses github-linguist to determine if a file is "good" or not
        """
        vendored = False
        generated = False
        try:
            query = ["github-linguist", file_name, "-j"]
            response = subprocess.run(
                " ".join(query),
                shell=True,
                capture_output=True,
                text=True,
                cwd=repo_dir,
            )
            result = json.loads(response.stdout)
            type = result[file_name]["type"]
            vendored = result[file_name]['vendored']
            generated = result[file_name]['generated']
            language = result[file_name]["language"]
            # if there is a string of numbers in the file name that is more than 4 characters long, it is likely autogenerated
            if vendored:
                return True, "This file is likely a vendored file."
            if generated:
                return True, "This file is likely an autogenerated file."
            if type != "Text":
                return True, "This file is likely not a code file."
            if language in self.excluded_languages:
                return True, f"This language for this file: {language} is usually not associated with coding."
            if language is None:
                return True, "A valid programming language could not be determined for this file."
            pattern = r'\d{5,}'
            match = re.search(pattern, file_name)
            if bool(match):
                return True, "The filename means that this file is likely auto generated."
        except Exception as e:
            logger.error(f"Error when checking if file is autogenerated: {e}")
            posthog.capture(
                "is_file_auto_generated_or_vendored", 
                "is_file_auto_generated_or_vendored error", 
                properties={"error": str(e), "file_name": file_name}
            )
        return False, ""
        

