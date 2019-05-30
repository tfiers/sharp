from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from sharp.config.types import ConfigDict, OneOrMoreLuigiTasks, RecordingFileID


CONFIG_FILENAME = "config.py"


@dataclass
class SharpConfig:
    # (Instruction for the PyCharm code editor, who can't handle dataclass
    # docstring yet):
    # noinspection PyUnresolvedReferences
    """
    :param get_tasks:  Return instantiated tasks, which will be passed to
            luigi.build(). The necessary import statements should be contained
            in this method's body (not at the top of the config.py file). This
            avoids circular imports (see config/README.md).
    
    :param raw_data:  A list of RecordingFileIDs.
    :param output_dir:  Path to a directory where the code may store processed
            data and output figures. Absolute path, or path relative to the
            config dir (where your "config.py" resides).
    :param shared_output_dir: Absolute path to a directory where the code may
            store processed data that is shared between different run
            configurations; i.e. data for which these different "config.py"
            files have the same values for all relevant options.
    
    :param fs_target:  Target sampling frequency after downsampling. In hertz.
    
    :param bitmap_versions:  If True, save PNG versions of figures, in addition
            to the PDF versions.
            
    :param logging:  A logging configuration passed to logging.dictConfig.
    
    :param scheduler_url:  Hostname where the remote luigi task scheduler
            is running. Only necessary when running multiple workers in
            parallel.
    :param config_id:  This setting allows to run multiple pipelines (each with
            a different config.py file) in parallel. Each such pipeline / config
            file corresponds to a different `config_id`. Default: name of parent
            directory of "config.py" file.

    :param mult_detect_ripple
    :param mult_detect_SW
    
    :param lockout_time:  In seconds.

    :param num_thresholds: Event detectioms are threshold crossings of an
            algorithm's output envelope, given that a certain "lockout" time
            has passed after the previous detection. This lockout time is based
            on the durations of all SWR events, namely the given percentile of
            durations. See ThresholdSweep.at_recall()
    
    :param train_fraction: Border between training and testing data, as a
            fraction of total signal duration.
    :param train_first:  Whether the training data comes before the test data or
            not.
    
    :param eval_start_extension:  How many seconds to extend the leading edge of
            reference SWR segments with when evaluating detections. Allows
            early detections (i.e. shortly before the reference segment starts)
            to count as correct detections.
    :param num_layers:  .. for the RNN
    :param num_units_per_layer
    :param chunk_duration:  Length of a chunk used in RNN training, in seconds.
            Network weights are updated after each chunk of training data has
            been processed.
    :param p_dropout:  Probability that a random hidden unit's activation is set
            to 0 during a training step. Should improve generalisation
            performance. Only relevant for num_layers > = 2.
    :param num_epochs:  How many times to pass over the training data when
            training an RNN.
    :param valid_fraction:  How much of the training data to use for validation
            (estimation of generalisation performance -- to choose net of epoch
            where this was maximal). The rest of the data is used for training
            proper.
    :param pos_weight:  When calculating the cost function for training the RNN,
            weight applied to positive training samples, i.e. where the desired
            output is "1" (SWR present). `pos_weight > 1` increases the recall,
            `pos_weight < 1` increases the precision.
    :param target_fullrect:  Shape of target function. Either variable-width
            rectangles on full reference segments (target_fullrect == True), or
            a fixed-width rectangle at refseg_start + [-target_start_pre,
            +target_start_post].
    :param target_start_pre
    :param target_start_post
    :param reference_seg_extension:  Reference segments are expanded at their
            leading edge, by the given fraction of total segment duration (=
            approximate SWR duration) before calculating the target signal.
            This should encourage SWR 'prediction' in the optimisation
            procedure.
    """

    get_tasks: Callable[[], OneOrMoreLuigiTasks]
    raw_data: Sequence[RecordingFileID]
    output_dir: str
    shared_output_dir: str
    fs_target: float
    bitmap_versions: bool
    logging: ConfigDict
    scheduler_url: Optional[str]
    config_id: str
    mult_detect_ripple: Tuple[float, ...]
    mult_detect_SW: Tuple[float, ...]
    lockout_time: float
    num_thresholds: int
    train_fraction: float
    train_first: bool
    eval_start_extension: float
    num_layers: int
    num_units_per_layer: int
    chunk_duration: float
    p_dropout: float
    num_epochs: int
    valid_fraction: float
    pos_weight: float
    target_fullrect: bool
    target_start_pre: float
    target_start_post: float
    reference_seg_extension: float
