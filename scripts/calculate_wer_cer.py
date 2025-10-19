import editdistance
import hydra
from pathlib import Path

def calc_cer(target_text: str, predicted_text: str) -> float:
    """
    Compute character error rate (CER).
    """
    if(not len(target_text)):
        return 1.0
    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text: str, predicted_text: str) -> float:
    """
    Compute word error rate (WER).
    """
    if(not len(target_text)):
        return 1.0
    return editdistance.eval(target_text.split(), predicted_text.split()) / len(
        target_text.split()
    )
@hydra.main(
    version_base=None, config_path="../src/configs", config_name="calc_metrics"
)
def main(config):
    prediction_dir = Path(config.prediction_path)
    ground_truth__dir = Path(config.ground_truth_path)
    cer=0
    wer=0
    count=0
    for gr_truth_f in ground_truth__dir.iterdir():
        if gr_truth_f.suffix!=".txt":
            continue
        count+=1
        file_name=gr_truth_f.name
        pred_f=prediction_dir/file_name
        if(not pred_f.exists()):
            continue
        pred_line=""
        with open(pred_f) as f:
            for line in f:
                pred_line+=line.rstrip("\n").lower().strip()
        gr_truth_line=""
        with open(gr_truth_f) as f:
            for line in f:
                gr_truth_line+=line.rstrip("\n").lower().strip()
        cer+=calc_cer(pred_line, gr_truth_line)
        wer+=calc_wer(pred_line, gr_truth_line)
    print(f"CER Mean is {cer/count} and WER Mean is {wer/count}")
if __name__=="__main__":
    main()