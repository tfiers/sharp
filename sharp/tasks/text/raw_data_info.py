from logging import getLogger
from os import stat
from os.path import exists
from pprint import pformat

from sharp.config.load import config, final_output_dir
from sharp.data.files.stdlib import DictFile
from sharp.tasks.base import SharpTask

log = getLogger(__name__)


def file_size(path: str) -> str:
    if exists(path):
        bytes = stat(path).st_size
        return f"{bytes / 1E9:.1f} GB ({path})"
    else:
        return f"NOPE ({path})"


class PrintFileSizes(SharpTask):

    out_file = DictFile(final_output_dir, "file sizes")

    def output(self):
        return self.out_file

    def work(self):
        data = {
            str(rec_file): file_size(rec_file.path)
            for rec_file in config.raw_data_paths
        }
        log.info(pformat(data))
        self.out_file.write(data)
