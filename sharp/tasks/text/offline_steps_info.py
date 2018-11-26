from numpy import diff, mean, std
from scipy.signal import kaiserord

from sharp.config.load import final_output_dir
from sharp.data.files.stdlib import DictFile
from sharp.tasks.base import SharpTask
from sharp.tasks.signal.base import InputDataMixin


class WriteOfflineInfo(SharpTask, InputDataMixin):
    def requires(self):
        return self.input_data_makers

    def output(self):
        return DictFile(final_output_dir, "offline-steps-info")

    def work(self):
        info_dict = {**self._filter_info(), **self._threshold_info()}
        self.output().write(info_dict)

    def _filter_info(self):
        rm = self.reference_maker
        fs = self.reference_channel_full.fs
        fn = fs / 2
        band = rm.band
        bw = diff(band)
        tw_fraction = from_percentage(rm.filter_options["transition_width"])
        transition_width = bw * tw_fraction
        attn = rm.filter_options["attenuation"]
        N, beta = kaiserord(attn, transition_width / fn)
        return {"Filter order N": N, "Beta": beta}

    def _threshold_info(self):
        rm = self.reference_maker
        median = float(rm.envelope_median)
        T_high = float(rm.threshold_high)
        T_low = float(rm.threshold_low)
        env_mean = float(mean(rm.envelope))
        env_std = float(std(rm.envelope))
        beta_high = (T_high - env_mean) / env_std
        beta_low = (T_low - env_mean) / env_std
        return {
            "Units": "microvolts",
            "T_high": T_high,
            "T_low": T_low,
            "Median": median,
            "Mean": env_mean,
            "Std": env_std,
            "beta_high": beta_high,
            "beta_low": beta_low,
        }


def from_percentage(pct: str) -> float:
    return float(pct.rstrip("%")) / 100
