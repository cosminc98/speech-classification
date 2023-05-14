import librosa
from typing import Optional, List, Dict
from scipy.stats import skew
import pandas as pd
import numpy as np
import random

from utils import AudioFile, SAMPLE_RATE

N_MFCC = 30


def stack_stats(
    features: np.ndarray, 
    axis: Optional[int] = None
) -> np.ndarray:
    
    return np.hstack(
        (
            np.mean(features, axis=axis), 
            np.std(features, axis=axis), 
            skew(features, axis=axis), 
            np.max(features, axis=axis), 
            np.median(features, axis=axis), 
            np.min(features, axis=axis)
        )
    )


def get_mfcc(audio_fpath: str, n_mfcc: int = 30, n_seconds: float = 1.0) -> Optional[np.ndarray]:
    data, _ = librosa.core.load(audio_fpath, sr=SAMPLE_RATE)

    n_samples = len(data)
    required_samples = int(SAMPLE_RATE * n_seconds)
    if n_samples != required_samples:
        if n_samples < required_samples:
            pad_samples = required_samples - n_samples
            pad_left = pad_samples // 2
            pad_right = pad_samples - pad_left
            data = np.pad(data, (pad_left, pad_right), "constant", constant_values=(0, 0))
        else:
            n_extra_samples = n_samples - required_samples
            data = data[n_extra_samples // 2 : n_extra_samples // 2 + required_samples]


    ft1 = librosa.feature.mfcc(y=data, sr=SAMPLE_RATE, n_mfcc=n_mfcc)
    ft2 = librosa.feature.zero_crossing_rate(y=data)[0]
    ft3 = librosa.feature.spectral_rolloff(y=data)[0]
    ft4 = librosa.feature.spectral_centroid(y=data)[0]
    ft5 = librosa.feature.spectral_contrast(y=data)[0]
    ft6 = librosa.feature.spectral_bandwidth(y=data)[0]

    ft1_stats = stack_stats(ft1, axis=1)
    ft2_stats = stack_stats(ft2)
    ft3_stats = stack_stats(ft3)
    ft4_stats = stack_stats(ft4)
    ft5_stats = stack_stats(ft5)
    ft6_stats = stack_stats(ft6)

    result = np.hstack(
        (ft1_stats, ft2_stats, ft3_stats, ft4_stats, ft5_stats, ft6_stats)
    )

    if np.isnan(np.min(result)):
        return None

    return result


def extract_features(
    audio_files: List[AudioFile],
    label_to_id: Optional[Dict[str, int]],
    subset: str, 
    n_mfcc: int = 30,
    n_seconds: float = 1.0,
):

    features: List[np.ndarray] = []
    labels: List[int] = []

    audio_files = [af for af in audio_files if af.subset == subset]

    for audio_file in audio_files:
        fts = get_mfcc(audio_file.file_path, n_seconds=n_seconds)

        if fts is None:
            continue

        features.append(fts)
        
        if label_to_id is not None:
            labels.append(label_to_id[audio_file.label])

    features: np.ndarray = np.stack(features)
    labels: np.ndarray = np.array(labels)

    return features, labels
