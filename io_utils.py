import os
import shutil
import json


def list_image_files(directory, exts=('png', 'jpg', 'jpeg')):
    """
    List all image files in a directory with given extensions.
    Returns sorted full paths.
    """
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(tuple(exts))
    )


def move_file(src_path: str, dest_dir: str):
    """
    Move a file to dest_dir, creating the directory if needed.
    """
    os.makedirs(dest_dir, exist_ok=True)
    shutil.move(src_path, os.path.join(dest_dir, os.path.basename(src_path)))


def save_json(obj: dict, path: str):
    """
    Dump a Python dict to a JSON file with indentation.
    """
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> dict:
    """
    Load JSON file into a Python dict.
    """
    with open(path) as f:
        return json.load(f)


def save_label_data(label_data: dict, base_path: str):
    """
    Save label_data as both JSON and simple TXT alongside base_path.
    base_path should exclude extension.
    """
    json_path = base_path + '.json'
    txt_path = base_path + '.txt'
    save_json(label_data, json_path)

    # Write a simple text representation for manual editing
    lines = []
    for key in ['filename', 'drop_count', 'xp_values', 'skills', 'boxes']:
        val = label_data.get(key)
        lines.append(f"{key}: {json.dumps(val)}")
    with open(txt_path, 'w') as f:
        f.write("".join(lines))


def load_label_data(base_path: str) -> dict:
    """
    Load label data, preferring JSON but falling back to TXT.
    """
    json_path = base_path + '.json'
    if os.path.exists(json_path):
        return load_json(json_path)

    txt_path = base_path + '.txt'
    data = {}
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            for line in f:
                if ': ' in line:
                    key, val = line.strip().split(': ', 1)
                    try:
                        data[key] = json.loads(val)
                    except json.JSONDecodeError:
                        data[key] = val
    return data