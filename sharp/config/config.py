from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from farao import Config
from sharp.config.fklab_data import fklab_data


@dataclass
class SharpConfig(Config):

    output_root: Path = "output/"

    raw_data: Mapping[str, Path] = field(default_factory=lambda: fklab_data)
    # A list of paths to ".raw.kwd", ".dat", and ".moz" files containing raw
    # neural recordings, indexed by arbitrary ID strings.

    fs_target: float = 1000
    # Target sampling frequency after downsampling. In hertz.
