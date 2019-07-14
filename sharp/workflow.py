from itertools import product

from sharp.datatypes.raw import RawRecordingFile
from sharp.init import config, sharp_workflow
from sharp.tasks.evaluation import (
    distill_performance_matrix,
    score_online_envelope,
)
from sharp.tasks.offline_analysis import (
    calc_SWR_segments,
    calc_mountain_heights,
    calc_pairwise_channel_differences,
    calc_ripple_envelope,
    calc_sharpwave_envelope,
    detect_mountains,
)
from sharp.tasks.online_detectors.RNN import (
    calc_RNN_envelope,
    get_init_model,
    initial_test_RNN_performance,
    select_RNN,
    tune_RNN_one_epoch,
)
from sharp.tasks.online_detectors.ripple_filter import (
    calc_online_ripple_filter_envelope,
)
from sharp.tasks.plot import (
    plot_F_matrix,
    plot_PR_curve,
    plot_multirec_F_matrix,
)
from sharp.tasks.preprocess import (
    downsample_raw,
    select_channel,
    split_segmentarray,
    split_signal,
    trim_recording,
)


def preprocess(raw_path, recording_ID):
    recording_file = RawRecordingFile.subclass_from(raw_path)
    downsampled_rec = downsample_raw(recording_file)
    ripple_env = calc_ripple_envelope(downsampled_rec)
    ripple_segs = detect_mountains(ripple_env)
    trimmed_recording = trim_recording(downsampled_rec, ripple_segs)
    return trimmed_recording


def select_best_channel(multichannel_envelope):
    segs = detect_mountains(multichannel_envelope)
    seg_heights = calc_mountain_heights(multichannel_envelope, segs)
    channel = select_channel(segs, seg_heights)
    return channel


def get_best_RNN(LFP_train, ref_SWR_train):
    # Divide training set even further for RNN training:
    LFP_tune, LFP_select = split_signal(
        LFP_train, config.RNN_training.tune_fraction
    )
    ref_SWR_tune, ref_SWR_select = split_segmentarray(
        ref_SWR_train, LFP_train, config.RNN_training.tune_fraction
    )
    model_performances = []
    model = get_init_model(LFP_tune)
    # Iterate over the tuning dataset several times
    for epoch in range(config.RNN_training.num_epochs):
        model = tune_RNN_one_epoch(model, LFP_tune, ref_SWR_tune)
        perf = initial_test_RNN_performance(model, LFP_select, ref_SWR_select)
        model_performances.append((model, perf))
    best_RNN = select_RNN(model_performances)
    return best_RNN


def compose_workflow():
    perf_matrices = []
    # continue: state directory

    for recording_ID, raw_recording_path in config.raw_data.items():
        LFP = preprocess(raw_recording_path, recording_ID)
        ripple_env = calc_ripple_envelope(LFP)
        pairwise_channel_diffs = calc_pairwise_channel_differences(LFP)
        sharpwave_env = calc_sharpwave_envelope(pairwise_channel_diffs)
        ripple_channel = select_best_channel(ripple_env)
        sharpwave_channel = select_best_channel(sharpwave_env)

        # Split data into training set and eventual reporting set.
        LFP_train, LFP_report = split_signal(
            full=LFP, fraction=config.train_fraction
        )
        # ORF = online ripple filter
        ORF_envelope_report = calc_online_ripple_filter_envelope(
            LFP_report, ripple_channel
        )
        detect_mult_matrix = product(
            config.ripple_detect_multipliers,
            config.sharpwave_detect_multipliers,
        )

        ORF_performances = []
        RNN_performances = []
        for ripple_detect_mult, sharpwave_detect_mult in detect_mult_matrix:
            reference_SWR_segs = calc_SWR_segments(
                ripple_env,
                ripple_channel,
                sharpwave_env,
                sharpwave_channel,
                ripple_detect_mult,
                sharpwave_detect_mult,
            )
            ref_SWR_train, ref_SWR_report = split_segmentarray(
                reference_SWR_segs, LFP, config.train_fraction
            )
            ORF_perf_report = score_online_envelope(
                ORF_envelope_report, ref_SWR_report
            )
            plot_PR_curve(ORF_perf_report)
            ORF_performances.append(ORF_perf_report)
            model = get_best_RNN(LFP_train, ref_SWR_train)
            RNN_envelope_report = calc_RNN_envelope(
                model, LFP_report, ref_SWR_report
            )
            RNN_perf_report = score_online_envelope(
                RNN_envelope_report, ref_SWR_report
            )
            plot_PR_curve(RNN_perf_report)
            RNN_performances.append(RNN_perf_report)

        perf_matrix = distill_performance_matrix(
            detect_mult_matrix, ORF_performances, RNN_performances
        )
        plot_F_matrix(perf_matrix)
        perf_matrices.append(perf_matrix)

    plot_multirec_F_matrix(perf_matrices)
