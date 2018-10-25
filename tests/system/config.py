from sharp.config.spec import SharpConfigBase


class SharpConfig(SharpConfigBase):

    # [Data]
    output_dir = "data/processed"
    raw_data_dir = "data/raw"
    reference_channel = "L2 - E13_extract"

    # [Main]
    config_id = "test"
    num_thresholds = 6

    # [NeuralNet]
    num_layers = 1
    num_units_per_layer = 25
    num_epochs = 2
