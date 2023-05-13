from dataclasses import dataclass
from typing import Optional

SAMPLE_RATE = 16000


@dataclass
class AudioFile:
    # path to the audio file
    file_path: str

    # the (speaker_id, utterance_id) pair uniquely identifies any file
    speaker_id: str
    utterance_id: str

    # for training, providing a label is required
    label: Optional[str] = None

    subset: str = "prediction"