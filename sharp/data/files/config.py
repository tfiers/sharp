from pathlib import Path

from luigi import BoolParameter, Config, FloatParameter, Parameter


class Data(Config):
    """
    At instantiation, reads in parameter values from the `[Data]`
    section of the `luigi.toml` file.
    """

    raw_data_dir = Parameter()
    # Directory containing raw NeuraLynx recordings (*.ncs files).

    output_dir = Parameter()
    # Path to a directory where the code may store processed data and output
    # figures. (Absolute path, or path relative to where the code is run from).

    bitmap_versions = BoolParameter(False)
    # If True, save bitmap copies of figures, next to the PDF versions.

    reference_channel = Parameter()
    # Name of the NCS file (without extension) that will be used for
    # single-channel filters and for defining reference SWR segments.

    train_fraction = FloatParameter(0.9)
    # Border between training and testing data, as a fraction of total signal
    # duration.
    train_first = BoolParameter(False)
    # Whether the training data comes before the test data or not.

    # Useful for choosing split boundary: relative timestamps of Kloosterman
    # Lab scientists labelling L2 data ('labelface' web app):
    #  - common set last event = 161 / 2040 = 0.0789
    #  - last labeller last event = 860 / 2040 = 0.4216


# Global config objects, for use in any Task.
data_config = Data()
output_root = Path(data_config.output_dir).absolute()
