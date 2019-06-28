from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from farao import Config


@dataclass
class SharpConfig(Config):

    raw_data: Mapping[str, Path]
    # A list of paths to ".raw.kwd", ".dat", and ".moz" files containing raw
    # neural recordings, indexed by arbitrary ID strings.

    fs_target: float
    # Target sampling frequency after downsampling. In hertz.
