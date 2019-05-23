from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sharp.config.types import RecordingFileID


# `frozen` makes this class hashable; i.e. usable as dict key.
@dataclass(frozen=True)
class RawDataPathPart:
    ID: Any
    partial_path: str = ""


class Rat(RawDataPathPart):
    ID: int


class Day(RawDataPathPart):
    ID: int


class File(RawDataPathPart):
    ID: str


def P1_dat_file(date: str, probe_name: str) -> File:
    return File(
        ID=probe_name,
        partial_path=f"{probe_name}/RatP1_{date}_{probe_name}/Klusta_raw_7-3.5/{probe_name}.dat",
    )


def P2_dat_file(probe_name: str) -> File:
    return File(
        ID=probe_name,
        partial_path=f"{probe_name}/Klusta_T7-3.5/{probe_name}.dat",
    )


FRED = "/mnt/nerffs01/ratlab/Frederic/organized/Long Flexible Probes/For the paper/data"
JJ = "/mnt/nerffs01/ratlab/Projects/Subiculum/Raw"


raw_data_paths = {
    Rat(1, f"{FRED}/P1/"): {
        Day(1): (
            P1_dat_file("16052014", "D1"),
            P1_dat_file("16052014", "D4"),
            P1_dat_file("16052014", "D10"),
            P1_dat_file("16052014", "D13"),
        ),
        Day(2): (
            P1_dat_file("18052014", "D1"),
            P1_dat_file("18052014", "D4"),
            P1_dat_file("18052014", "D10"),
            P1_dat_file("18052014", "D13"),
        ),
        Day(3): (
            P1_dat_file("20052014", "D1"),
            P1_dat_file("20052014", "D4"),
            P1_dat_file("20052014", "D10"),
            P1_dat_file("20052014", "D13"),
        ),
        Day(4): (
            P1_dat_file("22052014", "D1"),
            P1_dat_file("22052014", "D4"),
            P1_dat_file("22052014", "D10"),
            P1_dat_file("22052014", "D13"),
        ),
        Day(5): (
            P1_dat_file("24052014", "D1"),
            P1_dat_file("24052014", "D4"),
            P1_dat_file("24052014", "D10"),
            P1_dat_file("24052014", "D13"),
        ),
        Day(6): (
            P1_dat_file("26052014", "D1"),
            P1_dat_file("26052014", "D4"),
            P1_dat_file("26052014", "D10"),
            P1_dat_file("26052014", "D13"),
        ),
    },
    Rat(2): {
        Day(1, f"{FRED}/P2/28_08_2014/Nlx/"): (
            # P2_dat_file("D24"),  # No .dat file
            P2_dat_file("D27"),
            P2_dat_file("L1"),
            P2_dat_file("D21"),
            P2_dat_file("D26"),
            P2_dat_file("D30"),
            P2_dat_file("L5"),
            P2_dat_file("D29"),
            P2_dat_file("D22"),
            P2_dat_file("D23"),
            P2_dat_file("L4"),
            P2_dat_file("L2"),
            P2_dat_file("D15"),
        ),
        Day(2, f"{FRED}/P2/30_08_2014/Nlx/"): (
            # P2_dat_file("D24"),
            P2_dat_file("D27"),
            P2_dat_file("L1"),
            P2_dat_file("D21"),
            P2_dat_file("D26"),
            P2_dat_file("D30"),
            P2_dat_file("L5"),
            P2_dat_file("D29"),
            P2_dat_file("D22"),
            P2_dat_file("D23"),
            P2_dat_file("L4"),
            P2_dat_file("L2"),
            P2_dat_file("D15"),
        ),
        Day(3, f"{FRED}/P2/31_08_2014/"): (
            # P2_dat_file("D24"),
            P2_dat_file("D27"),
            P2_dat_file("L1"),
            P2_dat_file("D21"),
            P2_dat_file("D26"),
            P2_dat_file("D30"),
            P2_dat_file("L5"),
            P2_dat_file("D29"),
            P2_dat_file("D22"),
            P2_dat_file("D23"),
            P2_dat_file("L4"),
            P2_dat_file("L2"),
            P2_dat_file("D15"),
        ),
        Day(4, f"{JJ}/S001E000/2014-09-05_14-09-04/FP/"): (
            # File("D24", "p1.moz"),  # Error opening w/ NlxOpen
            File("D27", "p2.moz"),
            File("L1", "p3.moz"),
            File("D21", "p4.moz"),
            # File("D26", "p5.moz"),  # Not there
            # File("D30", "p6.moz"),  # Not there
            File("L5", "p9.moz"),
            File("D29", "p10.moz"),
            File("D22", "p12.moz"),
            File("D23", "p13.moz"),
            File("L4", "p14.moz"),
            File("L2", "p15.moz"),
            # File("D15", "p16.moz"),  # Contains NaN's
        ),
        Day(5, f"{JJ}/S001E000/2014-09-11_16-41-18/FP/"): (
            # File("D24", "p1.moz"),  # Contains NaN's (idem for the other 4 of this day)
            File("D27", "p2.moz"),
            File("L1", "p3.moz"),
            # File("D21", "p4.moz"),
            File("D26", "p5.moz"),
            File("D30", "p6.moz"),
            File("L5", "p9.moz"),
            # File("D29", "p10.moz"),
            File("D22", "p12.moz"),
            File("D23", "p13.moz"),
            File("L4", "p14.moz"),
            # File("L2", "p15.moz"),
            # File("D15", "p16.moz"),
        ),
    },
    Rat(3, f"{JJ}/S001E015/"): {
        Day(1, "2015-12-15_14-52-03/FP/"): (
            File("2 probes", "experiment0_100.raw.kwd"),
        )
    },
    Rat(4, f"{JJ}/S001E021/"): {
        Day(1, "2016-12-14_14-50-36/FP/"): (
            File("1 good probe", "experiment1_100.raw.kwd"),
        )
    },
}


flat_recordings_list = tuple(
    RecordingFileID(
        rat.ID,
        day.ID,
        file.ID,
        path=Path(rat.partial_path)
        / Path(day.partial_path)
        / Path(file.partial_path),
    )
    for rat, day_dict in raw_data_paths.items()
    for day, files in day_dict.items()
    for file in files
)
