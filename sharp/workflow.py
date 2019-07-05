from itertools import product

from sharp.data.files.raw import RawRecordingFile
from sharp.startup import config
from sharp.tasks.offline_analysis import (
    calc_SWR_segments,
    calc_mountain_heights,
    calc_pairwise_channel_differences,
    calc_ripple_envelope,
    calc_sharpwave_envelope,
    detect_mountains,
)
from sharp.tasks.online_detectors import (
    calc_SOTA_output_envelope,
    train_RNN_one_epoch,
)
from sharp.tasks.preprocess import (
    downsample_raw,
    select_channel,
    trim_recording,
)


def preprocess(raw_path, recording_ID):
    recording_file = RawRecordingFile.subclass_from(raw_path)
    downsampled_rec = downsample_raw(recording_file, output_name=recording_ID)
    ripple_env = calc_ripple_envelope(downsampled_rec)
    ripple_segs = detect_mountains(ripple_env)
    trimmed_recording = trim_recording([downsampled_rec, ripple_segs])
    return trimmed_recording


def select_best_channel(multichannel_envelope):
    segs = detect_mountains(multichannel_envelope)
    seg_heights = calc_mountain_heights([multichannel_envelope, segs])
    channel = select_channel([segs, seg_heights])
    return channel


for recording_ID, raw_recording_path in config.raw_data.items():
    trimmed_rec = preprocess(raw_recording_path, recording_ID)
    ripple_env = calc_ripple_envelope(trimmed_rec)
    pairwise_channel_diffs = calc_pairwise_channel_differences(trimmed_rec)
    sharpwave_env = calc_sharpwave_envelope(pairwise_channel_diffs)
    ripple_channel = select_best_channel(ripple_env)
    sharpwave_channel = select_best_channel(sharpwave_env)
    param_matrix = product(config.mult_detect_ripple, config.mult_detect_SW)
    SOTA_online_output_envelope = calc_SOTA_output_envelope(
        [trimmed_rec, ripple_channel]
    )
    for mult_detect_ripple, mult_detect_SW in param_matrix:
        reference_SWR_segs = calc_SWR_segments(
            ripple_env, ripple_channel, sharpwave_env, sharpwave_channel,
            mult_detect_ripple=mult_detect_ripple,
            mult_detect_SW=mult_detect_SW,
        )
        RNNs = []
        RNN = None
        for epoch in range(config.num_epochs):
            RNN = train_RNN_one_epoch([RNN, trimmed_rec, reference_SWR_segs])


if __name__ == "__main__":
    main()


# Continue: ..
# - convert all luigi tasks to functions + workflow call
# - run on test data
# - run sequentially with manual job on cluster :)
# - while that's running: make airflow dag, install postgres and run airflow svcs
