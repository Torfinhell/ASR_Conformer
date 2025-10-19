from src.text_encoder import CTCTextEncoder
from src.datasets import LibrispeechDataset 
BPE_DATASETS=["train-clean-100"]
def get_texts_for_bpe():
    texts=[]
    for dataset_name in BPE_DATASETS:
        dataset=LibrispeechDataset(dataset_name, text_encoder=CTCTextEncoder())
        texts += [dataset.get_text(ind) for ind in range(len(dataset))]
    return texts
