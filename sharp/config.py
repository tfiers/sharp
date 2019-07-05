import preludio

preludio.preload_with_feedback(
    ["scipy.signal", "fklab.segments", "matplotlib.pyplot"]
)


from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, Collection

from numpy import linspace
import farao

from sharp.OLD_datatypes.neuralnet import SharpRNN
from sharp.fklab_data import fklab_probe_recordings


@dataclass
class TrainingConfig:
    # Number of passes over the training data. Determines training duration.
    num_epochs: int = 60


class SharpConfig(farao.Config):
    # By default, save output files relative to where the workflow is run from.
    output_root: Path = "output/"

    # A collection of paths to ".raw.kwd", ".dat", and ".moz" files containing
    # raw neural recordings, each identified (for short reference) by an
    # arbitrary ID string.
    raw_data: Collection[Tuple[str, Path]] = fklab_probe_recordings

    # Target sampling frequency after downsampling. In hertz.
    fs_target: float = 1000

    mult_detect_ripple: Sequence[float] = tuple(linspace(1, 3, 0.5))
    mult_detect_SW: Sequence[float] = tuple(linspace(1, 3, 0.5))

    SharpRNN_config: SharpRNN.Config = SharpRNN.Config()
    training_config: TrainingConfig = TrainingConfig()


config = SharpConfig.load()
sharp_workflow = farao.Workflow(config)

