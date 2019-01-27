from sharp.config.load import config
from sharp.data.types.split import TrainTestSplit
from sharp.tasks.plot.base import TimeRangesPlotter
from sharp.tasks.plot.util.annotations import add_segments
from sharp.tasks.signal.reference import MakeReference


class PlotReferenceMaker(TimeRangesPlotter):
    """
    Plots the input signal, the reference maker filter output, and the
    reference segments. Does this only for the evaluation slice.
    """

    selected_time_ranges_only = False
    reference_channel_only = False
    full_range_scalebars = True
    output_dir = TimeRangesPlotter.output_dir / "reference"
    window_size = 2
    figwidth = 1

    reference_makers = [
        MakeReference(**args) for args in config.make_reference_args
    ]
    rm0 = reference_makers[0]

    def requires(self):
        return super().requires() + tuple(self.reference_makers)

    @property
    def extra_signals(self):
        envelopes = (
            self.rm0.ripple_envelope,
            self.rm0.SW_envelope,
            self.rm0.toppyr_envelope,
            self.rm0.sr_envelope,
        )
        return [TrainTestSplit(env).signal_test for env in envelopes]

    def post_plot(self, time_range, input_ax, extra_axes):
        self.add_SWR_segs(input_ax)
        self.add_SW_segs(extra_axes[1])
        self.add_ripple_segs(extra_axes[0])
        # ax = extra_axes[0]
        # ax.hlines(self.reference_maker.ripple_threshold_high, *time_range)
        # ax.hlines(
        #     self.reference_maker.ripple_threshold_low,
        #     *time_range,
        #     linestyles="dashed"
        # )

    def add_SWR_segs(self, ax):
        for rm in self.reference_makers:
            self.add_segs(ax, rm.output().read())

    def add_SW_segs(self, ax):
        mr = config.mult_detect_ripple[-1]
        for ms in config.mult_detect_SW:
            rm = MakeReference(mult_detect_SW=ms, mult_detect_ripple=mr)
            self.add_segs(ax, rm.calc_SW_segments())

    def add_ripple_segs(self, ax):
        ms = config.mult_detect_SW[-1]
        for mr in config.mult_detect_ripple:
            rm = MakeReference(mult_detect_SW=ms, mult_detect_ripple=mr)
            self.add_segs(ax, rm.calc_ripple_segments())

    def add_segs(self, ax, segs):
        env = self.rm0.ripple_envelope
        segs_test = TrainTestSplit(env, segs).segments_test
        add_segments(ax, segs_test, alpha=0.1)
