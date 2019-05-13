from pathlib import Path

from sharp.config.spec import SharpConfigBase
from sharp.config.types import RecordingFileID


class SharpConfig(SharpConfigBase):

    # [Data]
    raw_data_paths = (
        RecordingFileID(
            rat=2, day=5, probe="D29", path=Path("data/raw/sig.moz")
        ),
        RecordingFileID(
            rat=2, day=2, probe="L2", path=Path("data/raw/sig.dat")
        ),
        RecordingFileID(
            rat=3, day=1, probe="Waluigi", path=Path("data/raw/sig.raw.kwd")
        ),
    )
    output_dir = "data/processed"
    shared_output_dir = "data/processed-shared"
    reference_channel = "L2 - E13_extract"
    toppyr_channel_ix = 2
    sr_channel_ix = 0

    # [Main]
    config_id = "test"
    num_thresholds = 6
    channel_combinations = {
        "all": (0, 1, 2),
        "pyr": (0,),
        "sr": (2,),
        "pyr+sr": (0, 2),
    }
    time_ranges = [(0.68, 1.2)]

    # [NeuralNet]
    num_layers = 1
    num_units_per_layer = 8
    num_epochs = 2
    RNN_channel_combo_name = "pyr"
