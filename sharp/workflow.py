from sharp.util.startup import preload

preload()


from itertools import product
from farao import Scheduler
from sharp.config.config import SharpConfig
from sharp.data.files.raw import RawRecordingFile
from sharp.tasks.online_detectors import calc_SOTA_output_envelope, \
    train_RNN_one_epoch
from sharp.tasks.preprocess import (
    downsample_raw,
    select_channel,
    trim_recording,
)
from sharp.tasks.offline_analysis import (
    calc_ripple_envelope,
    detect_mountains,
    calc_mountain_heights,
    calc_pairwise_channel_differences,
    calc_sharpwave_envelope,
    calc_SWR_segments,
)


config = SharpConfig.load_from_directory()
schedule = Scheduler(config)


def preprocess(raw_path, recording_ID):
    recording_file = RawRecordingFile.subclass_from(raw_path)
    downsampled_rec = schedule(
        downsample_raw, input=recording_file, output_name=recording_ID
    )
    ripple_env = schedule(calc_ripple_envelope, input=downsampled_rec)
    ripple_segs = schedule(detect_mountains, input=ripple_env)
    trimmed_recording = schedule(
        trim_recording, input=[downsampled_rec, ripple_segs]
    )
    return trimmed_recording


def select_best_channel(multichannel_envelope):
    segs = schedule(detect_mountains, input=multichannel_envelope)
    seg_heights = schedule(
        calc_mountain_heights, input=[multichannel_envelope, segs]
    )
    channel = schedule(select_channel, input=[segs, seg_heights])
    return channel


for recording_ID, raw_recording_path in config.raw_data.items():
    trimmed_rec = preprocess(raw_recording_path, recording_ID)
    ripple_env = schedule(calc_ripple_envelope, input=trimmed_rec)
    pairwise_channel_diffs = schedule(
        calc_pairwise_channel_differences, input=trimmed_rec
    )
    sharpwave_env = schedule(
        calc_sharpwave_envelope, input=pairwise_channel_diffs
    )
    ripple_channel = select_best_channel(ripple_env)
    sharpwave_channel = select_best_channel(sharpwave_env)
    param_matrix = product(config.mult_detect_ripple, config.mult_detect_SW)
    SOTA_online_output_envelope = schedule(
        calc_SOTA_output_envelope, input=[trimmed_rec, ripple_channel]
    )
    for mult_detect_ripple, mult_detect_SW in param_matrix:
        reference_SWR_segs = schedule(
            calc_SWR_segments,
            input=[
                ripple_env,
                ripple_channel,
                sharpwave_env,
                sharpwave_channel,
            ],
            mult_detect_ripple=mult_detect_ripple,
            mult_detect_SW=mult_detect_SW,
        )
        RNNs = []
        RNN = None
        for epoch in range(config.num_epochs):
            RNN = schedule(train_RNN_one_epoch, input=[RNN, trimmed_rec, reference_SWR_segs])


def main():
    schedule.run_sequentially()


if __name__ == "__main__":
    main()

# Continue: ..
# - convert all luigi tasks to functions + workflow schedule call
# - run on test data
# - run sequentially with manual job on cluster :)
# - while that's running: make airflow dag, install postgres and run airflow svcs
