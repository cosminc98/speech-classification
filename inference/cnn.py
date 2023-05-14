import os
import random
import shutil
import tempfile
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torchvision import datasets, transforms
from utils import AudioFile, SAMPLE_RATE


class Net(nn.Module):   
    def __init__(self, n_classes: int, use_last_maxpool: bool):
        super(Net, self).__init__()

        self.n_classes = n_classes
        self.use_last_maxpool = use_last_maxpool

        # input size (201, 161)
        cnn_layers = []
        cnn_layers.append(nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2))  # (201, 161)
        cnn_layers.append(nn.BatchNorm2d(64))
        cnn_layers.append(nn.ReLU(inplace=True))
        cnn_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # (100, 80)
        cnn_layers.append(nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2))  # (100, 80)
        cnn_layers.append(nn.BatchNorm2d(128))
        cnn_layers.append(nn.ReLU(inplace=True))
        if self.use_last_maxpool:
            cnn_layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) # (50, 40)
        cnn_layers.append(nn.Dropout2d(p=0.5))
        self.cnn_layers = nn.Sequential(*cnn_layers)

        if self.use_last_maxpool:
            fc_size = 128 * 50 * 40
        else:
            fc_size = 128 * 100 * 80

        self.linear_layers = nn.Sequential(
            nn.Linear(fc_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes),
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return F.log_softmax(x, dim=1)
    

def load_cnn_model(model_fpath: str, n_classes: int, use_large_model: bool):
    model = Net(n_classes, use_last_maxpool=(not use_large_model))
    model.load_state_dict(torch.load(model_fpath, map_location=torch.device('cpu')))
    model.eval()
    return model
    

def create_spectrogram_images_dataset(
    audio_files: List[AudioFile],
    dataset_dirpath: str,
    eps: float = 0.0000001,
    force_overwrite: bool = False,
    verbose = False,
) -> None:

    if os.path.exists(dataset_dirpath):
        if force_overwrite:
            shutil.rmtree(dataset_dirpath)
        else:
            if verbose:
                print(f'"{dataset_dirpath}" already exists. Skipping generating spectrograms.')
            return

    if not os.path.exists(dataset_dirpath):
        os.makedirs(dataset_dirpath)

    resamplers: Dict[int, T.Resample] = {}

    for audio_file in audio_files:
        waveform, sample_rate = torchaudio.load(audio_file.file_path)

        if sample_rate not in resamplers:
            resamplers[sample_rate] = T.Resample(sample_rate, SAMPLE_RATE, dtype=torch.float32)

        # if sample_rate != SAMPLE_RATE:
        #     raise Exception("Wrong sampling rate.")

        waveform = resamplers[sample_rate](waveform)

        N_SECONDS = 2.0

        n_samples = len(waveform[0])
        required_samples = int(SAMPLE_RATE * N_SECONDS)
        if n_samples != required_samples:
            if n_samples < required_samples:
                pad_samples = required_samples - n_samples
                pad_left = pad_samples // 2
                pad_right = pad_samples - pad_left
                waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right))
            else:
                n_extra_samples = n_samples - required_samples
                waveform = waveform[:, n_extra_samples // 2 : n_extra_samples // 2 + required_samples]

        # add a small value to the spectrogram to prevent -inf values when doing
        # .log2 on the plt.imsave method below
        spectrogram_tensor = torchaudio.transforms.Spectrogram()(waveform) + eps
        
        fig = plt.figure()
        png_fpath = os.path.join(
            dataset_dirpath, 
            audio_file.subset, 
            audio_file.label, 
            f"spec_img_{audio_file.speaker_id}_{audio_file.utterance_id}.png"
        )
        
        dir_path = os.path.dirname(png_fpath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        plt.imsave(
            fname=png_fpath, 
            arr=spectrogram_tensor[0].log2()[:,:].numpy(), 
            cmap='viridis'
        )
        plt.close()

    if verbose:
        print(f"Dataset {os.path.basename(dataset_dirpath)}:")
        for subset in os.listdir(dataset_dirpath):
            subset_dirpath = os.path.join(dataset_dirpath, subset)
            for label in os.listdir(subset_dirpath):
                label_dirpath = os.path.join(subset_dirpath, label)
                print(f'\t"{subset}" subset, "{label}" label: {len(os.listdir(label_dirpath))} samples')


def load_spectrogram_images(spectrograms_path: str, subset: str):
    subset_path = os.path.join(spectrograms_path, subset)

    if not os.path.exists(subset_path):
       return None

    spec_dataset = datasets.ImageFolder(
        root=subset_path,
        transform=transforms.Compose(
            [   
                transforms.Resize((201,161)),
                transforms.ToTensor()
            ]
        )
    )
    return spec_dataset


def predict_cnn(audio_fpath: str, model: nn.Module, id_to_label: Dict[int, str], device=torch.device('cpu')):
    audio_file = AudioFile(
        file_path=audio_fpath, 
        speaker_id="UNKNOWN", 
        utterance_id="predict_sample", 
        subset="predict", 
        label="UNKNOWN"
    )
    
    temp_dir = tempfile.TemporaryDirectory()
    tmp_dirpath = temp_dir.name

    create_spectrogram_images_dataset(
        audio_files=[audio_file], 
        dataset_dirpath=tmp_dirpath, 
        force_overwrite=True
    )
    dataset_predict = load_spectrogram_images(tmp_dirpath, "predict")
    dataloader_predict = torch.utils.data.DataLoader(
        dataset_predict,
        batch_size=1,
        num_workers=1,
        shuffle=False,
    )

    model.eval()

    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader_predict):
            X, Y = X.to(device), Y.to(device)
            logits = model(X)

            probabilities = F.softmax(logits, dim=1).cpu()

            predicted_label_id = int(logits[0].argmax(0))
            predicted_label_name = id_to_label[predicted_label_id]

            print(
                f'Predicted label "{predicted_label_name.upper()}" '
                f"with {torch.max(probabilities) * 100:.2f}% confidence"
            )

            msg = "\t("
            for class_idx in range(len(id_to_label)):
                msg += f"{id_to_label[class_idx]}: {float(probabilities[0][class_idx]) * 100:.2f}%"
                if class_idx != len(id_to_label) - 1:
                    msg += "; "
                else:
                    msg += ")"
            print(msg)

            break

    temp_dir.cleanup()

    return predicted_label_name