from logging import getLogger
from os import stat
from os.path import exists
from pprint import pformat

from numpy import nan, nansum

from sharp.config.load import config, final_output_dir
from sharp.data.files.stdlib import DictFile
from sharp.tasks.base import SharpTask

log = getLogger(__name__)


def file_size(path: str) -> float:
    """ In gigabytes. NaN if file does not exist. """
    if exists(path):
        bytes = stat(path).st_size
        return bytes / 1e9
    else:
        return nan


class PrintFileSizes(SharpTask):

    out_file = DictFile(final_output_dir, "file sizes")

    def output(self):
        return self.out_file

    def work(self):
        rec_files = config.raw_data_paths
        file_sizes = [file_size(r.path) for r in rec_files]
        size_strings = [
            f"{size:.1f} GB" if size is not nan else "file not found"
            for size in file_sizes
        ]
        output = {
            str(rec_file): f"{size} ({rec_file.path})"
            for rec_file, size in zip(rec_files, size_strings)
        }
        output["Total"] = f"{nansum(file_sizes):.1f} GB"
        log.info(pformat(output))
        self.out_file.write(output)
