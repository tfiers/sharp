from sharp.util.startup import preload


preload()

from pathlib import Path
from farao import Scheduler
from sharp.config.config import SharpConfig
from sharp.data.files.raw import RawRecordingFile
from sharp.tasks.preprocess.downsample import downsample_raw
from sharp.tasks.preprocess.filter import (
    calc_ripple_envelope,
    calc_sharpwave_envelope,
)


config = SharpConfig.load_from_directory()
schedule = Scheduler(config)

output_root = Path(config.output_root)
preprocess = output_root / "preprocess"

for recording_ID, path in config.raw_data.items():
    recording_file = RawRecordingFile.subclass_from(path)
    downsampled_rec = schedule(
        downsample_raw, input=recording_file, output_name=recording_ID
    )
    ripple_env = schedule(calc_ripple_envelope, input=downsampled_rec)
    sharpwave_env = schedule(calc_sharpwave_envelope, input=downsampled_rec)


schedule.run_sequentially()

# Continue: ..
# - convert all luigi tasks to functions + workflow schedule call
# - run on test data
# - run sequentially with manual job on cluster :)
# - while that's running: make airflow dag, install postgres and run airflow svcs
