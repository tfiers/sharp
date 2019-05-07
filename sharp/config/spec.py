from itertools import product
from os import environ
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from warnings import warn

from numpy import linspace
from typeguard import check_type

from sharp.config.default.channels import (
    L2_channel_combinations,
    L2_channels,
    L_probe_outline,
)
from sharp.config.default.logging import LOGGING_CONFIG
from sharp.config.default.raw_data import flat_recordings_list
from sharp.data.types.config import ConfigError, RecordingFile

CONFIG_DIR_ENV_VAR = "SHARP_CONFIG_DIR"
config_dir = Path(environ.get(CONFIG_DIR_ENV_VAR, ".")).expanduser().absolute()
ConfigDict = Dict[str, Union[Any, "ConfigDict"]]

# We do not want to import from luigi yet. (As it executes initalization code on
# import. We want to control this initialization by setting env vars, later).
# Therefore make a dummy Luigi.Task type.
LuigiTask = TypeVar("LuigiTask")

# A semantic alias for `NotImplemented`
MANDATORY_SETTING = NotImplemented


class SharpConfigBase:
    def get_tasks(self) -> Union[LuigiTask, Iterable[LuigiTask]]:
        """
        Return instantiated tasks, which will be passed to luigi.build().
        The necessary import statements should be contained in this method's
        body (not at the top of the config.py file). This avoids circular
        imports (see config/README.md).
        
        Developer note: should not be called before the `sharp.config.load`
        script has run (e.g. after an object from it is imported).
        """
        import sharp.config.default.tasks as default_tasks

        return default_tasks.tasks_to_run

    #
    # Data settings
    # -------------

    raw_data_paths: Sequence[RecordingFile] = flat_recordings_list

    raw_data_dir: str = "obsolete"
    # Directory containing raw NeuraLynx recordings (*.ncs files).

    output_dir: str = "output"  # MANDATORY_SETTING
    # Path to a directory where the code may store processed data and output
    # figures. (Absolute path, or path relative to your custom `config.py`
    # file; i.e. relative to the "sharp config dir" env var).

    reference_channel: str = "L2 - E13"  # MANDATORY_SETTING
    # Name of the NCS file (without extension) that will be used for
    # single-channel detection algorithms and for defining reference SWR
    # segments.

    bitmap_versions: bool = False
    # If True, save PNG versions of figures, in addition to the PDF versions.

    toppyr_channel_ix: int = 15
    sr_channel_ix: int = 6
    # These channels are used for offline sharp wave detection

    #
    # Logging and Luigi worker config
    # -----------------

    logging: ConfigDict = LOGGING_CONFIG

    luigi_scheduler_host: str = "nerfcluster-fs"
    # Hostname where the remote luigi task scheduler is running (useful when
    # running multiple workers in parallel).

    #
    # Main settings
    # -----------------

    config_id: Optional[str] = None
    # This setting allows to run multiple pipelines (each with a different
    # config.py file) in parallel. Each such pipeline / config file corresponds
    # to a different `config_id`. By default, takes the value of the "sharp
    # config dir" env var.

    mult_detect_ripple = tuple(linspace(0.4, 4, num=7))
    mult_detect_SW = tuple(linspace(0.9, 5, num=7))
    make_reference_args = [
        dict(mult_detect_SW=mult_SW, mult_detect_ripple=mult_ripple)
        for mult_SW, mult_ripple in product(mult_detect_SW, mult_detect_ripple)
    ]

    channel_combinations: Dict[str, Sequence[int]] = L2_channel_combinations

    probe_outline: Sequence[Tuple[float, float]] = L_probe_outline
    electrodes_x: Sequence[float] = [ch.x for ch in L2_channels]
    electrodes_y: Sequence[float] = [ch.y for ch in L2_channels]

    # lockout_time: float = 34e-3
    lockout_time: float = 60e-3
    # In seconds. Based on the 25-percentile refseg length lockout of earlier.

    # lockout_percentile: float = 25
    # Event detectios are threshold crossings of an algorithm's output
    # envelope, given that a certain "lockout" time has passed after the
    # previous detection. This lockout time is based on the durations of all
    # SWR events, namely the given percentile of durations.

    num_thresholds: int = 64

    selected_recall: float = 0.8
    # See ThresholdSweep.at_recall()

    train_fraction: float = 0.6
    # Border between training and testing data, as a fraction of total signal
    # duration.
    train_first: bool = True
    # Whether the training data comes before the test data or not.

    # Useful for choosing split boundary: relative timestamps of Kloosterman
    # Lab scientists labelling L2 data ('labelface' web app):
    #  - common set last event = 161 / 2040 = 0.0789
    #  - last labeller last event = 860 / 2040 = 0.4216

    time_ranges: Sequence[Tuple[float, float]] = [
        (107.2, 107.8),
        # (107.69, 107.79),  # Zoom-in
        (107.33, 107.45),  # Zoom-in
        (349.2, 349.8),  # Clean & strong ripples
        (132.6, 133.2),  # Lotsa ripply & merging
        # from plot/paper/signals.py:
        (19.8, 20.8),
        (606.4, 607.5),
        (552.6, 553.2),  # early rnn ok visible. But one late SW
        (369.14, 369.65),  # three clean SPWR, early SW. But RNN not convincing
        (183.06, 184.5),
        (343.36, 344.3),  # nah, too weak SWs
        (183.09, 183.84),  # nice early RNN. Take me.
    ]
    # Segments of data to use for time-range plots. In seconds, relative to the
    # start of the evaluation (AKA test) slice.

    eval_start_extension: float = 14 / 1000
    # How many seconds to extend the leading edge of reference SWR segments
    # with when evaluating detections. Allows early detections (i.e. shortly
    # before the reference segment starts) to count as correct detections.

    #
    # RNN architecture
    # ----------------
    num_layers: int = 2
    num_units_per_layer: int = 40
    RNN_channel_combo_name: str = "all"

    #
    # RNN training settings
    # -----------------

    chunk_duration: float = 0.3
    # Length of a chunk, in seconds. Network weights are updated after each
    # chunk of training data has been processed.

    p_dropout: float = 0.4
    # Probability that a random hidden unit's activation is set to 0 during a
    # training step. Should improve generalisation performance. Only relevant
    # for num_layers >= 2.

    num_epochs: int = 15
    # How many times to pass over the training data when training an RNN.

    valid_fraction: float = 0.22
    # How much of the training data to use for validation (estimation of
    # generalisation performance -- to choose net of epoch where this was
    # maximal). The rest of the data is used for training proper.

    pos_weight: float = 1.0
    # When calculating the cost function for training the RNN, weight applied
    # to positive training samples, i.e. where the desired output is "1" (SWR
    # present). `pos_weight > 1` increases the recall, `pos_weight < 1`
    # increases the precision.

    target_fullrect: bool = False
    target_start_pre: float = 14 / 1000
    target_start_post: float = 25 / 1000
    # Shape of target function. Either binary on full reference segments,
    # or a rectangle at refseg_start + [-pre, +post].

    reference_seg_extension: float = 0
    # Reference segments are expanded at their leading edge, by the given
    # fraction of total segment duration (= approximate SWR duration) before
    # calculating the target signal. This should encourage SWR 'prediction' in
    # the optimisation procedure.

    #
    # Internals
    # ---------

    def __init__(self) -> None:
        self._validate()
        self._normalize()

    def _validate(self):
        settings = dir(SharpConfigBase)
        for name in dir(self):
            if name not in settings:
                warn(f"SharpConfig attribute `{name}` is not a config setting.")
            value = getattr(self, name)
            if value == NotImplemented:
                raise ConfigError(
                    f"The mandatory config setting `{name}` is not set."
                )
            expected_type = self.__annotations__.get(name)
            if expected_type:
                try:
                    check_type(name, value, expected_type)
                except TypeError as e:
                    raise ConfigError(
                        f"The config setting `{name}` has an incorrect type. "
                        f"Expected type: `{expected_type}`. "
                        f"See preceding exception for details."
                    ) from e

    def _normalize(self):
        if self.config_id is None:
            self.config_id = str(config_dir)

        def _as_absolute_Path(path: str) -> Path:
            ppath = Path(path)
            if ppath.is_absolute():
                return ppath
            else:
                return config_dir / ppath

        self.output_dir = _as_absolute_Path(self.output_dir)
        self.raw_data_dir = _as_absolute_Path(self.raw_data_dir)

    def get_tasks_tuple(self) -> Tuple[LuigiTask, ...]:
        tasks = self.get_tasks()
        try:
            iter(tasks)
            return tuple(tasks)
        except TypeError:
            return (tasks,)
