import os


def get_deepest_folders(directory):
    deepest_folders = []
    for root, dirs, files in os.walk(directory):
        if not dirs and '61' in root:  # No subdirectories in this folder
            deepest_folders.append(root)
    return deepest_folders


# Example usage
directory = 'saved_models/2.0'
deepest_folders = get_deepest_folders(directory)
for folder in deepest_folders:
    print('"'+folder+'"')
