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
    get_init_RNN,
    calc_mean_sample_loss,
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
    split_segmentlist,
    split_signal,
    trim_recording,
)

# continue: make runnable in tests/system

rec_dirs = []

for recording_ID, abs_path in raw_data:

    # A separate directory for each neural recording
    rec_dir = workflow_outputs.mkdir(recording_ID)
    rec_dirs.append(rec_dir)

    # We will mainly be working with only a part of the full recording. Put the
    # full recording in a subdir.
    full_rec_dir = rec_dir.mkdir("full_recording")

    full_rec_dir["downsampled_recording"] = (downsample_raw, (abs_path))
    full_rec_dir["ripple_envelope"] = (
        calc_ripple_envelope,
        (full_rec_dir["downsampled_recording"]),
    )
    full_rec_dir["ripple_segs"] = (
        detect_mountains,
        (full_rec_dir["ripple_envelope"]),
    )

    # The part we will be working on.
    rec_dir["LFP"] = (
        trim_recording,
        (full_rec_dir["downsampled_rec"], full_rec_dir["ripple_segs"]),
    )
    rec_dir["ripple_envelope"] = (calc_ripple_envelope, (rec_dir["LFP"]))
    rec_dir["pairwise_channel_diffs"] = (
        calc_pairwise_channel_differences,
        (rec_dir["LFP"]),
    )
    rec_dir["sharpwave_envelope"] = (
        calc_sharpwave_envelope,
        (rec_dir["pairwise_channel_diffs"]),
    )

    # Select the channel with the strongest ripples. Idem for sharpwaves.
    for motif in ("ripple", "sharpwave"):
        select_ch_dir = rec_dir.mkdir(f"select_{motif}_channel")
        select_ch_dir["segs"] = (
            detect_mountains,
            (rec_dir[f"{motif}_envelope"]),
        )
        select_ch_dir["seg_heights"] = (
            calc_mountain_heights,
            (select_ch_dir["segs"]),
        )
        rec_dir[f"{motif}_channel"] = (
            select_channel,
            (select_ch_dir["seg_heights"]),
        )

    # Split data into training set and eventual reporting set.
    rec_dir[("LFP__train", "LFP__report")] = (
        split_signal,
        (rec_dir["LFP"], config.train_fraction),
    )

    # SOTA (state of the art) = online ripple filter
    rec_dir["SOTA_envelope__report"] = (
        calc_online_ripple_filter_envelope,
        (rec_dir["LFP_report"], rec_dir["ripple_channel"]),
    )

    # Divide training set even further for later RNN training:
    rec_dir[("LFP__tune", "LFP__select")] = (
        split_signal,
        (rec_dir["LFP__train"], config.RNN_training.tune_fraction),
    )

    # The following analyses will each use a different pair of offline ripple
    # and sharpwave detection thresholds. Each analysis / threshold pair has
    # its own directory. The following directory is a container for all these
    # directories.
    detection_thresholds_dir = rec_dir.mkdir("detection_thresholds")
    detection_threshold_matrix = product(
        config.ripple_detect_multipliers, config.sharpwave_detect_multipliers
    )

    thr_pair_dirs = []
    for thr_pair in detection_threshold_matrix:
        ripple_detection_threshold, sharpwave_detection_threshold = thr_pair
        dirname = (
            f"R_{ripple_detection_threshold:.1f}___"
            f"SW_{sharpwave_detection_threshold:.1f}"
        ).replace(".", "_")
        # R_1_0___SW_2_5
        thr_pair_dir = detection_thresholds_dir.mkdir(dirname)
        thr_pair_dirs.append(thr_pair_dir)

        # This pair of detection thresholds yields a specific set of reference
        # sharpwave-ripple segments.
        thr_pair_dir["reference_SWR_segs"] = (
            calc_SWR_segments,
            (
                rec_dir["ripple_envelope"],
                rec_dir["ripple_channel"],
                rec_dir["sharpwave_envelope"],
                rec_dir["sharpwave_channel"],
                ripple_detection_threshold,
                sharpwave_detection_threshold,
            ),
        )

        # Split these segments into a training and an eventual reporting set.
        thr_pair_dir[("ref_SWR__train", "ref_SWR__report")] = (
            split_segmentlist,
            (
                thr_pair_dir["reference_SWR_segs"],
                rec_dir["LFP"],
                config.train_fraction,
            ),
        )

        # Use the reporting set to evaluate the SOTA algorithm performance for
        # the current detection threshold pair.
        thr_pair_dir["SOTA_performance"] = (
            score_online_envelope,
            (rec_dir["SOTA_envelope__report"], thr_pair_dir["ref_SWR__report"]),
        )
        thr_pair_dir["SOTA_PR_curve"] = (
            plot_PR_curve,
            (thr_pair_dir["SOTA_performance"]),
        )

        # Like the training LFP input data, also split the training reference
        # segments furhter:
        thr_pair_dir[("ref_SWR__tune", "ref_SWR__select")] = (
            split_segmentlist,
            (
                thr_pair_dir["ref_SWR__train"],
                rec_dir["LFP__train"],
                config.RNN_training.tune_fraction,
            ),
        )

        # Next, we train a recurrent neural network (RNN) for multiple
        # iterations over the tuning data.
        RNN_dir = thr_pair_dir.mkdir("RNNs")
        for i in range(config.RNN_training.num_epochs):
            if i == 0:
                RNN_dir[f"model_{i}"] = (get_init_RNN, (rec_dir["LFP__tune"]))
            else:
                RNN_dir[f"model_{i}"] = (
                    tune_RNN_one_epoch,
                    (
                        RNN_dir[f"model_{i-1}"],
                        rec_dir["LFP__tune"],
                        thr_pair_dir["ref_SWR__tune"],
                    ),
                )
            RNN_dir[f"model_{i}_performance"] = (
                calc_mean_sample_loss,
                rec_dir["LFP__select"],
                thr_pair_dir["ref_SWR__select"],
            )
            # Alex: use crossvalidation here, instead of only one split.

        # Select the iteration that yielded the best performance on the
        # training data held-out from RNN tuning.
        thr_pair_dir["best_RNN"] = (
            select_RNN,
            (
                [
                    (RNN_dir[f"model_{i}"], RNN_dir[f"model_{i}_performance"])
                    for i in range(config.RNN_training.num_epochs)
                ]
            ),
        )
        # note: fileflow needs to recursively expand arguments, and replace
        # every object of type `output_subdir["filename"]`.

        # Run the chosen RNN on the reporting data.
        thr_pair_dir["RNN_envelope__report"] = (
            calc_RNN_envelope,
            (
                thr_pair_dir["best_RNN"],
                rec_dir["LFP__report"],
                thr_pair_dir["ref_SWR__report"],
            ),
        )
        # nice to have: don't require ref SWR here.
        # Just allow target = None in Input_TargetPair.

        # Like for the SOTA algorithm, use the reporting set to evaluate the
        # performance of the chosen RNN for the current detection threshold
        # pair:
        thr_pair_dir["RNN_performance"] = (
            score_online_envelope,
            (rec_dir["RNN_envelope__report"], thr_pair_dir["ref_SWR__report"]),
        )
        thr_pair_dir["RNN_PR_curve"] = (
            plot_PR_curve,
            (thr_pair_dir["RNN_performance"]),
        )

    rec_dir["performance_matrix"] = (
        distill_performance_matrix,
        (
            detection_threshold_matrix,
            [tp_dir["SOTA_performance"] for tp_dir in thr_pair_dirs],
            [tp_dir["RNN_performance"] for tp_dir in thr_pair_dirs],
        ),
    )
    rec_dir["F_matrix"] = (plot_F_matrix, (rec_dir["performance_matrix"]))


workflow_outputs["multirec_F_matrix"] = (
    plot_multirec_F_matrix,
    ([rec_dir["F_matrix"] for rec_dir in rec_dirs]),
)
