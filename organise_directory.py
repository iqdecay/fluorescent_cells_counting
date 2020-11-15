import os

cwd = os.getcwd()

for file in os.listdir(cwd):
    filename, extension = os.path.splitext(file)
    directory_name = filename
    if extension == ".czi" :
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        else :
            print(f"Directory {directory_name} already exists, skipping")
        old_name = file
        new_name = os.path.join(directory_name,file)
        os.rename(old_name, new_name)
        print("Created directory {directory_name} and moved corresponding .czi"
              " inside it")

