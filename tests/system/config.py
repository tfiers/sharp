from pathlib import Path

from sharp.config.spec import SharpConfig
from sharp.config.types import RecordingFileID


config = SharpConfig(
    # [Data]
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
    # [Main]
    config_id="test",
    num_thresholds=6,
    time_ranges=[(0.68, 1.2)],
    # [NeuralNet]
    num_layers=1,
    num_units_per_layer=8,
    num_epochs=2,
)
