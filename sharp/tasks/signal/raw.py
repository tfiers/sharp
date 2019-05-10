from sharp.config.load import config
from sharp.data.files.raw_data import (
    RawDATFile,
    RawKwikFile,
    RawRecording,
    TahitiFile,
)
from sharp.data.types.config import RecordingFileID
from sharp.tasks.base import CustomParameter, ExternalTask, SharpTask


class SingleRecordingFileTask(SharpTask):
    file_ID: RecordingFileID = CustomParameter()


class RawRecording_ExistenceCheck(ExternalTask, SingleRecordingFileTask):
    def output(self) -> RawRecording:
        path = self.file_ID.path
        extension = path.suffix.lower()
        possible_classes = (RawDATFile, RawKwikFile, TahitiFile)
        FileClass = next(
            C for C in possible_classes if C.extension == extension
        )
        return FileClass(
            directory=path.parent, filename=path.name.split(".")[0]
        )


class CheckWhetherAllRawRecordingsExist(SharpTask):
    def requires(self):
        return (
            RawRecording_ExistenceCheck(file_ID=rec_file)
            for rec_file in config.raw_data_paths
        )
