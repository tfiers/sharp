from typing import Dict, Optional, Tuple, TypeVar

from sharp.config.default.channels import L2_channel_combinations

MANDATORY_SETTING = NotImplemented

# We may not import from luigi yet: we want to set its env vars later.
LuigiTask = TypeVar("LuigiTask")


class SharpConfigBase:
    def get_tasks(self) -> Tuple[LuigiTask, ...]:
        """
        Return instantiated tasks, which will be passed to luigi.build().
        The necessary import statements should be contained in this method's
        body (not at the top of the config.py file). This avoids circular
        imports, as Tasks in sharp use config data.
        """
        import sharp.config.default.tasks as default_tasks

        return default_tasks.tasks_to_run

    # Data settings
    # -------------

    raw_data_dir: str = MANDATORY_SETTING
    # Directory containing raw NeuraLynx recordings (*.ncs files).

    output_dir: str = MANDATORY_SETTING
    # Path to a directory where the code may store processed data and output
    # figures. (Absolute path, or path relative to your custom `config.py`
    # file; i.e. relative to the "sharp config dir" env var).

    reference_channel: str = MANDATORY_SETTING
    # Name of the NCS file (without extension) that will be used for
    # single-channel detection algorithms and for defining reference SWR
    # segments.

    bitmap_versions: bool = False
    # If True, save PNG versions of figures, in addition to the PDF versions.

    # Main settings
    # -----------------

    config_id: Optional[str] = None
    # This setting allows to run multiple pipelines (each with a different
    # config file) in parallel. Each such pipeline / config file corresponds to
    # a different `config_id`. By default, takes the value of the
    # "sharp config dir" env var.

    channel_combinations: Dict[str, Tuple[int, ...]] = L2_channel_combinations

    lockout_percentile: float = 25
    # Event detectios are threshold crossings of an algorithm's output
    # envelope, given that a certain "lockout" time has passed after the
    # previous detection. This lockout time is based on the durations of all
    # SWR events, namely the given percentile of durations.

    num_thresholds: int = 64

    recall_best: float = 0.8
    # See ThresholdSweep.best()

    train_fraction: float = 0.6
    # Border between training and testing data, as a fraction of total signal
    # duration.
    train_first: bool = True
    # Whether the training data comes before the test data or not.

    # Useful for choosing split boundary: relative timestamps of Kloosterman
    # Lab scientists labelling L2 data ('labelface' web app):
    #  - common set last event = 161 / 2040 = 0.0789
    #  - last labeller last event = 860 / 2040 = 0.4216

    # RNN architecture
    # ----------------
    num_layers: int = 2
    num_units_per_layer: int = 20

    # RNN training settings
    # -----------------
    reference_seg_extension: float = 0
    # Reference segments are expanded at their leading edge, by the given
    # fraction of total segment duration (= approximate SWR duration). This
    # should encourage SWR 'prediction' in the optimisation procedure.

    chunk_duration: float = 0.3
    # Length of a chunk, in seconds. Network weights are updated after each
    # chunk of training data has been processed.

    p_dropout: float = 0.4
    # Probability that a random hidden unit's activation is set to 0 during a
    # training step. Should improve generalisation performance. Only relevant
    # for num_layers > 1.

    num_epochs: int = 10
    # How many times to pass over the training data when training an RNN.

    valid_fraction: float = 0.22
    # How much of the training data to use for validation (estimation of
    # generalisation performance -- to choose net of epoch where this was
    # best). The rest of the data is used for training proper.
