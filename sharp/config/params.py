"""
Each subclass of luigi.Config will read in parameter values from its
corresponding section in the `luigi.toml` file on instantiation.
(E.g. the [Data] section for `class Data(Config)`).
"""

from pathlib import Path

from luigi import BoolParameter, Config, FloatParameter, IntParameter
from luigi.task import Parameter


class Main(Config):
    config_id = Parameter()
    # Setting this to a custom value allows to run multiple pipelines (each
    # with a different `luigi.toml` config file) in parallel. Each such
    # pipeline / config file corresponds to a different `config_id`.

    lockout_percentile = FloatParameter(25)
    # Event detectios are threshold crossings of an algorithm's output envelope,
    # after a certain "lockout" time after the previous detection has passed.
    # This lockout time is based on the durations of all SWR events, namely
    # the given percentile of durations.

    num_thresholds = IntParameter(64)

    recall_best = FloatParameter(0.8)
    # See ThresholdSweep.best()


class Data(Config):

    raw_data_dir = Parameter()
    # Directory containing raw NeuraLynx recordings (*.ncs files).

    output_dir = Parameter()
    # Path to a directory where the code may store processed data and output
    # figures. (Absolute path, or path relative to where the code is run from).

    bitmap_versions = BoolParameter(False)
    # If True, save PNG versions of figures, in addition to the PDF versions.

    reference_channel = Parameter()
    # Name of the NCS file (without extension) that will be used for
    # single-channel detection algorithms and for defining reference SWR
    # segments.

    train_fraction = FloatParameter(0.6)
    # Border between training and testing data, as a fraction of total signal
    # duration.
    train_first = BoolParameter(True)
    # Whether the training data comes before the test data or not.

    # Useful for choosing split boundary: relative timestamps of Kloosterman
    # Lab scientists labelling L2 data ('labelface' web app):
    #  - common set last event = 161 / 2040 = 0.0789
    #  - last labeller last event = 860 / 2040 = 0.4216


class NeuralNet(Config):
    # RNN architecture
    # ----------------
    num_layers: int = IntParameter(2)
    num_units_per_layer: int = IntParameter(20)

    # Training settings
    # -----------------
    reference_seg_extension: float = FloatParameter(0)
    # Reference segments are expanded at their leading edge, by the given
    # fraction of total segment duration (= approximate SWR duration). This
    # should encourage SWR 'prediction' in the optimisation procedure.

    chunk_duration: float = FloatParameter(0.3)
    # Length of a chunk, in seconds. Network weights are updated after each
    # chunk of training data has been processed.

    p_dropout: float = FloatParameter(0.4)
    # Probability that a random hidden unit's activation is set to 0 during a
    # training step. Should improve generalisation performance. Only relevant
    # for num_layers > 1.

    num_epochs: int = IntParameter(10)
    # How many times to pass over the training data when training an RNN.

    valid_fraction: float = FloatParameter(0.22)
    # How much of the training data to use for validation (estimation of
    # generalisation performance -- to choose net of epoch where this was
    # best). The rest of the data is used for training proper.


# Global config objects, for use in any Task:

main_config = Main()
data_config = Data()
neural_net_config = NeuralNet()

output_root = Path(data_config.output_dir).absolute()
intermediate_output_dir = output_root / "intermediate"
final_output_dir = output_root / "final"
