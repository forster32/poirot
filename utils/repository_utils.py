import os

from config.client import PoirotConfig


class Repository:
    root_directory: str

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
