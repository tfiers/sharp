from sharp.config.spec import SharpConfigBase
from sharp.data.types.config import RecordingFile


class SharpConfig(SharpConfigBase):

    # [Data]
    raw_data_dir = "data/raw"
    raw_data_paths = (
        RecordingFile(rat=1, day=1, probe="Waluigi", path="test.kak"),
    )
    output_dir = "data/processed"
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
