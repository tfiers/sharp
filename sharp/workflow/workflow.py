from sharp.util.startup import preload

preload()

from farao import Scheduler, load_config
from sharp.config.spec import SharpConfig
from sharp.data.files.raw import RawRecordingFile
from sharp.tasks.downsample import downsample_raw


config: SharpConfig = load_config()
schedule = Scheduler(config)

for recording_ID, path in config.raw_data.items():
    recording_file = RawRecordingFile.subclass_from(path)
    downsampled_signal = schedule(
        downsample_raw, input=recording_file, output_name=recording_ID
    )

schedule.run_sequentially()

# Continue: ..
# - commit
# - see what's next for pretty figures
