from logging import getLogger
from random import random
from time import sleep

from sharp.config.types import RecordingFileID

log = getLogger(__name__)


def downsample(file_ID: RecordingFileID):
    print(f"print start downsample {file_ID}")
    sleep(2 + 20 * random())
    log.info("Down bye")
