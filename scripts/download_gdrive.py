from pathlib import Path
import gdown
import zipfile

GDRIVE_URLS = {
    "models": {
        "https://drive.google.com/uc?id=1a1gjSXB3mMsNOHcdndhJW6ABZ8HzBEFm": "data/models/chk_bpeablation.pth",  # bpeablation
        "https://drive.google.com/uc?id=1a1gjSXB3mMsNOHcdndhJW6ABZ8HzBEFm": "data/models/chk_train-clean-360.pth",  # train-clean-360
        "https://drive.google.com/uc?id=1yPsuK5BZahriXlwl87TqY6WUuxO5WSH1": "data/models/chk_train-other-500.pth"   # train-other-500
    },
    "dataset": {
        "https://drive.google.com/uc?id=1ZEqEX6s7lWvCqBRclKyRW4TwJQFcxH4F": "data/datasets/example"
    }
}

def download_models(gdrive_urls):
    if "models" not in gdrive_urls:
        raise ValueError("Cannot upload model files")
    path_gzip = Path("data/models").absolute()
    path_gzip.mkdir(exist_ok=True, parents=True)
    for url, path in gdrive_urls["models"].items():
        gdown.download(url, path)

def download_dataset(gdrive_urls):
    if "dataset" not in gdrive_urls:
        raise ValueError("Cannot upload dataset files")
    path_gzip = Path("data/datasets").absolute()
    path_gzip.mkdir(exist_ok=True, parents=True)
    for url, path in gdrive_urls["dataset"].items():
        zip_path = path + ".zip"
        gdown.download(url, zip_path)
        if zip_path.endswith('.zip'):
            extract_folder = path
            Path(extract_folder).mkdir(exist_ok=True, parents=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
            Path(zip_path).unlink()

if __name__ == "__main__":
    download_models(GDRIVE_URLS)
    download_dataset(GDRIVE_URLS)
