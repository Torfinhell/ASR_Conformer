import editdistance
import torch 
def calc_cer(target_text:str, pred_text:str):
    assert len(target_text)
    return editdistance.eval(target_text, pred_text)/len(target_text)
class ArgmaxCERMetric():
    def __init__(self, text_encoder):
        self.name="ARGMAX_CER"
        self.text_encoder=text_encoder
    def __call__(self, log_probs, log_probs_length, text, **batch):
        cers=[]
        predictions=torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths=log_probs_length.cpu().detach().numpy()
        for prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text=self.text_encoder.normalize(target_text)
            pred_text=self.text_encoder.ctc_decode(prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))   
        return sum(cers)/len(cers)