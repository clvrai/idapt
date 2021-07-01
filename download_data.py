import sys
import os

import gdown
import zipfile
import shutil
import os


DIR = "./data"

DEMOS = {
    "InvertedPendulum": "1XG37i09_8IwywHoT98CVnjIACj7K_bya",
    "HalfCheetah": "1X-s72B65WJ9V2fCj66IzXtGaWLT0l6vE",
    "Walker2d": "1epegZg0_tMY6sHe8KDmVyF3wI8SDUr5L",
    "FetchReach": "131C8f0nEzmy7y4oN4yVKbT4dkGlhUppB",
    "SawyerPush": "1xsnGDc5BVK58_okNsR0689gkwYBq1qTl",
}

FILENAMES = {
    "InvertedPendulum": "invertedpendulum",
    "HalfCheetah": "halfcheetah",
    "Walker2d": "walker2d",
    "FetchReach": "fetchreach",
    "SawyerPush": "sawyerpush",
}


if __name__ == "__main__":
    tasks = []
    if len(sys.argv) > 1:
        tasks = sys.argv[1:]
    else:
        tasks = [
            "InvertedPendulum",
            "HalfCheetah",
            "Walker2d",
            "FetchReach",
            "SawyerPush",
        ]

    os.makedirs(DIR, exist_ok=True)

    for task in tasks:
        id = DEMOS[task]
        url = "https://drive.google.com/uc?id=" + id
        target_path = "%s/%s.zip" % (DIR, task)

        print("Downloading demo %s from %s" % (task, url))
        gdown.download(url, target_path)

        with zipfile.ZipFile(target_path, "r") as zip_ref:
            zip_ref.extractall(DIR)
        os.remove(target_path)
        unzipped_path = "%s/%s" % (DIR, FILENAMES[task])
        files = os.listdir(unzipped_path)
        for file_name in files:
            shutil.move(os.path.join(unzipped_path, file_name), DIR)
        os.rmdir(unzipped_path)

