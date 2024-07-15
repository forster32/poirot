import os

from dataclasses import dataclass

from config.client import PoirotConfig
from config.server import ROOT_DIRECTORY

@dataclass
class Repository:
    root_directory=ROOT_DIRECTORY

    def get_file_list(self):
        files = []
        poirot_config: PoirotConfig = PoirotConfig()

        def dfs_helper(directory_path):
            nonlocal files
            for item in os.listdir(directory_path):
                if item == ".git":
                    continue
                if item in poirot_config.exclude_dirs:  # this saves a lot of time
                    continue
                item_path = os.path.join(directory_path, item)
                if os.path.isfile(item_path):
                    # make sure the item_path is not in one of the banned directories
                    if not poirot_config.is_file_excluded(item_path):
                        files.append(item_path)  # Add the file to the list
                elif os.path.isdir(item_path):
                    dfs_helper(item_path)  # Recursive call to explore subdirectory

        dfs_helper(self.root_directory)
        files = [file[len(self.root_directory) + 1 :] for file in files]
        return files
    

    def get_file_contents(self, file_path, ref=None):
        local_path = os.path.join(self.root_directory, file_path.lstrip("/"))
        if os.path.exists(local_path) and os.path.isfile(local_path):
            with open(local_path, "r", encoding="utf-8", errors="replace") as f:
                contents = f.read()
            return contents
        else:
            raise FileNotFoundError(f"{local_path} does not exist.")
    
    # TODO
    # def __del__(self):
