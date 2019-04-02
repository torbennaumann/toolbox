import os

def file_in_data(file_path):
    root_dir = os.path.dirname(__file__)

    return f'{root_dir}/../../data/{file_path}'

