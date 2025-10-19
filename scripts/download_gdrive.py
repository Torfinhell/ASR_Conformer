from pathlib import Path
import gdown
GDRIVE_URLS= {
    "https://drive.google.com/file/d/1a1gjSXB3mMsNOHcdndhJW6ABZ8HzBEFm/view?usp=drive_link": "data/models/model_best.pth",
    # "": "saved/train_other_500_2/model_best.pth",
}

def main():
    path_gzip=Path("data/models").absolute()
    path_gzip.mkdir(exist_ok=True, parents=True)
    for url, path in GDRIVE_URLS.items():
        gdown.download(url, path)
        
if __name__== "__main__":
    main()
