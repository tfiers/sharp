from logging import getLogger
from pprint import pformat

from numpy import sum

from sharp.config.load import config, final_output_dir
from sharp.data.files.stdlib import DictFile
from sharp.tasks.base import SharpTask
from sharp.tasks.signal.raw import CheckWhetherAllRawRecordingsExist

log = getLogger(__name__)


class PrintFileSizes(SharpTask):
    def requires(self):
        return CheckWhetherAllRawRecordingsExist()

    def output(self):
        return DictFile(final_output_dir, "file sizes")

    def work(self):
        rec_files = config.raw_data_paths
        file_sizes_GB = [r.path.stat().st_size / 1e9 for r in rec_files]
        out_dict = {
            f"{rec_file} ({rec_file.path})": f"{size:.1f} GB"
            for rec_file, size in zip(rec_files, file_sizes_GB)
        }
        out_dict["Total"] = f"{sum(file_sizes_GB):.1f} GB"
        log.info(pformat(out_dict))
        self.output().write(out_dict)
