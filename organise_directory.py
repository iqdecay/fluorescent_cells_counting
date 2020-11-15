import sys
import os

"""
This scripts takes a directory name as argument, and then for all `.czi` files
in it, creates a directory with the same name and moves the `.czi` file inside
it

Usage (for the current directory) :
    `python3 organise_directory.py .`
"""
directory = sys.argv[1]
if not os.path.exists(directory):
    raise NotADirectoryError(f"Provided argument {directory} isn't an existing directory")

for file in os.listdir(directory):
    filename, extension = os.path.splitext(file)
    directory_name = os.path.join(directory, filename)
    if extension == ".czi" :
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        else :
            print(f"Directory {directory_name} already exists, skipping")
        old_name = os.path.join(directory, file)
        new_name = os.path.join(directory_name,file)
        os.rename(old_name, new_name)
        print(f"Created directory {directory_name} and moved corresponding .czi"
              " inside it")

