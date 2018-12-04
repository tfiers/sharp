from abc import ABC, abstractmethod
from os import environ
from pathlib import Path
from textwrap import fill
from typing import Dict, Iterable, Optional, Sequence, Tuple, TypeVar, Union
from warnings import warn

from numpy import diff, array, ndarray
from typeguard import check_type

from sharp.config.default.channels import (
    L2_channel_combinations,
    L2_channels,
    L_probe_outline,
)

CONFIG_DIR_ENV_VAR = "SHARP_CONFIG_DIR"

config_dir = Path(environ.get(CONFIG_DIR_ENV_VAR, ".")).absolute()


# We do not want to import from luigi yet. (As it executes initalization code on
# import. We want to control this initialization by setting env vars, later).
# Therefore make a dummy Luigi.Task type.
LuigiTask = TypeVar("LuigiTask")

MANDATORY_SETTING = NotImplemented


class SharpConfigBase:
    def get_tasks(self) -> Union[LuigiTask, Iterable[LuigiTask]]:
        """
        Return instantiated tasks, which will be passed to luigi.build().
        The necessary import statements should be contained in this method's
        body (not at the top of the config.py file). This avoids circular
        imports, as Tasks in sharp use config data.
        
        Developer note: should not be called before the `sharp.config.load`
        script has run (e.g. after an object from it is imported).
        """
        import sharp.config.default.tasks as default_tasks

        return default_tasks.tasks_to_run

    #
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

    #
    # Main settings
    # -----------------

    config_id: Optional[str] = None
    # This setting allows to run multiple pipelines (each with a different
    # config file) in parallel. Each such pipeline / config file corresponds to
    # a different `config_id`. By default, takes the value of the
    # "sharp config dir" env var.

    channel_combinations: Dict[str, Sequence[int]] = L2_channel_combinations

    probe_outline: Sequence[Tuple[float, float]] = L_probe_outline
    electrodes_x: Sequence[float] = [ch.x for ch in L2_channels]
    electrodes_y: Sequence[float] = [ch.y for ch in L2_channels]

    lockout_percentile: float = 25
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

    offline_steps_segs: Sequence[Tuple[float, float]] = [(0.68, 1.2)]
    # Segment of data to use for the `offline SWR detection steps` plot. In
    # seconds, relative to the start of the evaluation (AKA test) slice.

    #
    # RNN architecture
    # ----------------
    num_layers: int = 2
    num_units_per_layer: int = 20

    #
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
    # at_recall). The rest of the data is used for training proper.

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


class ConfigError(Exception):
    """
    Raised when the environment is not configured properly to run `sharp`
    tasks.
    """

    def __init__(self, message: str):
        # Make sure the complete error message fits nice & square in the
        # terminal.
        future_prefix = f"{self.__class__}: "
        square = fill(future_prefix + message, width=80)
        square_with_prefix_sized_hole = square[len(future_prefix) :]
        super().__init__(square_with_prefix_sized_hole)


def num_taps_BPF(order: int) -> int:
    return 2 * order + 1


def num_delays_BPF(order: int) -> int:
    return num_taps_BPF(order) - 1


class LTIRippleFilter(ABC):

    passband = (100, 200)

    @abstractmethod
    def get_taps(self, order, fs) -> (ndarray, ndarray):
        """
        :param order:  Order N of a typical band-pass filter, created by
                    convolution of a low-pass and a high-pass filter. (I.e.
                    order N for which num_taps = 2 * N + 1).
        :param fs:  Signal sampling frequency, in Hz.
        :return: (b, a), i.e. coefficients of (numerator, denominator) of the
        filter.
        """

    @property
    def bandwidth(self):
        return diff(self.passband)

    def get_passband_normalized(self, fs):
        f_nyq = fs / 2
        return array(self.passband) / f_nyq

    def __repr__(self):
        return str(self.__class__.__name__)
