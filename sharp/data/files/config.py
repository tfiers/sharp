from pathlib import Path

from luigi import Config, IntParameter, Parameter


class DataConfig(Config):
    """
    At instantiation, reads in parameter values from the `[DataConfig]`
    section of the `luigi.toml` file.
    """

    raw_data_dir = Parameter()
    # Directory containing raw NeuraLynx recordings (*.ncs files).

    output_dir = Parameter()
    # Path to a directory where the code may store processed data and output
    # figures. (Absolute path, or path relative to where the code is run from).

    probe_number = IntParameter()

    electrode_number = IntParameter()


# Global config objects, for use in any Task.
data_config = DataConfig()
output_root = Path(data_config.output_dir).absolute()
