import sys
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from hydra.utils import instantiate
import hydra
from src.logger.utils import plot_spectrogram
from src.text_encoder import CTCTextEncoder
from src.transforms.spec_augs import MaskFreq, TimeMask
from src.transforms.wav_augs import Gain, ShiftPitch
from omegaconf import OmegaConf
from src.utils.init_utils import setup_saving_and_logging

@hydra.main(version_base=None, config_path="../src/configs", config_name="show_augs")
def main(config):
    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)
    dataset = instantiate(config.datasets["inference"], text_encoder=CTCTextEncoder())
    for i in range(min(config.trainer.show_examples, len(dataset))):
        writer.set_step(i, "")
        instance_data = dataset[i]
        spectrogram_current = instance_data["spectrogram"]
        audio = instance_data["audio"]
        sample_rate = config.trainer.sample_rate
        spectrogram_mask_freq = MaskFreq()(spectrogram_current.unsqueeze(0)).squeeze(0)
        spectrogram_time_mask = TimeMask()(spectrogram_current.unsqueeze(0)).squeeze(0)
        audio_pitch_shift = Gain()(audio)
        audio_gain = ShiftPitch(sample_rate=sample_rate)(audio)
        writer.add_audio("audio", audio, sample_rate=sample_rate)
        writer.add_audio("Gain", audio_gain, sample_rate=sample_rate)
        writer.add_audio("ShiftPitch", audio_pitch_shift, sample_rate=sample_rate)
        writer.add_image("MaskFreq", image_spectrogram(spectrogram_mask_freq, config))
        writer.add_image("spectrogram", image_spectrogram(spectrogram_current, config))
        writer.add_image("TimeMask", image_spectrogram(spectrogram_time_mask, config))


def image_spectrogram(spectrogram, config):
    image = plot_spectrogram(spectrogram, config)
    return image

if __name__ == "__main__":
    main()