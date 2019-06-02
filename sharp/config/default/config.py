# This "config.py" file was copied from "[source_code]/sharp/config/default/"
# to your custom sharp configuration directory. Change the argument values
# in the "SharpConfig()" initialisation call below to suit your needs.

from pathlib import Path

from numpy import linspace

from sharp.config.default.logging import get_logging_config
from sharp.config.default.raw_data import flat_recordings_list
from sharp.config.default.tasks import get_default_tasks
from sharp.config.spec import SharpConfig


config = SharpConfig(
    get_tasks=get_default_tasks,
    raw_data=flat_recordings_list,
    output_dir="./output",
    shared_output_dir="/home/ratlab/tomas/data/shared",
    fs_target=1000,
    bitmap_versions=False,
    logging=get_logging_config(multiple_workers=True),
    scheduler_url="http://nerfcluster-fs:8082",
    config_id=str(Path(__file__).parent.stem),
    mult_detect_ripple=tuple(linspace(0.4, 4, num=7)),
    mult_detect_SW=tuple(linspace(0.9, 5, num=7)),
    lockout_time=60e-3,
    num_thresholds=64,
    train_fraction=0.6,
    train_first=True,
    eval_start_extension=14e-3,
    num_layers=2,
    num_units_per_layer=40,
    chunk_duration=0.3,
    p_dropout=0.4,
    num_epochs=15,
    valid_fraction=0.22,
    pos_weight=1.0,
    target_fullrect=False,
    target_start_pre=14e-3,
    target_start_post=25e-3,
    reference_seg_extension=0,
    max_memory_usage=0.9,
)
