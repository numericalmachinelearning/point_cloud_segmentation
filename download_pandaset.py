"""
download_pandaset.py

Script to download the ENTIRE PandaSet dataset from Kaggle
and unzip it into a target folder.

Requirements:
    pip install kaggle python-dotenv

Create a .env file in this folder with:
    KAGGLE_USERNAME=your_username
    KAGGLE_KEY=your_api_key

Run:
    python download_pandaset.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv


DATASET_SLUG = "usharengaraju/pandaset-dataset"


def download_pandaset(target_dir: str = "./pandaset"):
    # Load env before importing Kaggle
    load_dotenv()

    from kaggle.api.kaggle_api_extended import KaggleApi

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not username or not key:
        raise RuntimeError("KAGGLE_USERNAME or KAGGLE_KEY missing in .env")

    api = KaggleApi()
    api.authenticate()

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    print(f"Downloading FULL PandaSet dataset into: {target.resolve()}")
    print("This is ~33GB. This may take a long time and a lot of disk space.\n")

    api.dataset_download_files(
        DATASET_SLUG,
        path=str(target),
        unzip=True,   # unzip after download
    )

    print("Download complete.")
    print("Contents are in:", target.resolve())


if __name__ == "__main__":
    download_pandaset("./pandaset")
