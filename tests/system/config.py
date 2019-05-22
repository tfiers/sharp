from pathlib import Path

from numpy import linspace
from sharp.config.default.logging import LOGGING_CONFIG
from sharp.config.default.tasks import get_default_tasks
from sharp.config.spec import SharpConfig
from sharp.config.types import RecordingFileID


config = SharpConfig(
    get_tasks=get_default_tasks,
    raw_data=(
        RecordingFileID(
            rat=2, day=5, probe="D29", path=Path("data/raw/sig.moz")
        ),
        RecordingFileID(
            rat=2, day=2, probe="L2", path=Path("data/raw/sig.dat")
        ),
        RecordingFileID(
            rat=3, day=1, probe="Waluigi", path=Path("data/raw/sig.raw.kwd")
        ),
    ),
    output_dir="data/processed",
    shared_output_dir="data/processed-shared",
    fs_target=1000,
    bitmap_versions=False,
    logging=LOGGING_CONFIG,
    scheduler_url=None,
    config_id="test",
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
)
