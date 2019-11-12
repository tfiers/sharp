from abc import ABC
from logging import getLogger

from sharp.config.load import config
from sharp.config.types import RecordingFileID
from sharp.data.files.raw_data import (
    RawDATFile,
    RawKwikFile,
    RawRecording,
    TahitiFile,
)
from sharp.tasks.base import CustomParameter, SharpTask, WrapperTask


log = getLogger(__name__)


class SingleRecordingFileTask(SharpTask, ABC):
    file_ID: RecordingFileID = CustomParameter()

    # To help code completion:
    def __init__(self, file_ID: RecordingFileID, **kwargs):
        super().__init__(file_ID=file_ID, **kwargs)


class RawRecording_ExistenceCheck(SingleRecordingFileTask, WrapperTask):
    def output(self) -> RawRecording:
        path = self.file_ID.path
        filename, extension = path.name.split(".", 1)
        extension = "." + extension
        possible_classes = (RawDATFile, RawKwikFile, TahitiFile)
        try:
            FileClass = next(
                C for C in possible_classes if C.extension == extension
            )
        except StopIteration:
            log.error(
                f"{self.file_ID} ({self.file_ID.path}) is not one of the"
                f" accepted file formats"
                f" ({', '.join(C.extension for C in possible_classes)})."
            )
        return FileClass(directory=path.parent, filename=filename)


class CheckWhetherAllRawRecordingsExist(WrapperTask):
    def requires(self):
        return (
            RawRecording_ExistenceCheck(file_ID=rec_file)
            for rec_file in config.raw_data
        )
