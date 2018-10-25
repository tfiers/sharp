from dataclasses import dataclass
from typing import Optional


@dataclass
class Channel:
    name: str
    index: int
    recording_site: Optional[str] = None


SO = "Stratum oriens"
Pyr = "Pyramidal cell layer"
SR = "Stratum radiatum"
SLM = "Stratum lacunosum-moleculare"

# `L2` is the L-style-probe recording at:
# >> nerffs01/ratlab/Frederic/organized/Long Flexible Probes/For the paper/data/P2/30_08_2014/Nlx/L2
# The recording sites are inaccurate guesses, based on the Kjonigsen composition
# I made.
_L2_channels = (
    Channel("linear-array, bottom", 0, SLM),
    Channel("linear-array, bottom+1", 1, SR),
    Channel("linear-array, bottom+2", 2, SR),
    Channel("linear-array, mid-low", 3, SR),
    Channel("linear-array, mid-high", 4, SR),
    Channel("linear-array, top-2", 5, SR),
    Channel("linear-array, top-1", 6, SR),
    Channel("linear-array, top", 7, Pyr),
    Channel("cluster, bottom, L", 8, Pyr),
    Channel("cluster, bottom, R", 9, Pyr),
    Channel("cluster, mid-low, L", 10, Pyr),
    Channel("cluster, mid-low, R", 11, Pyr),
    Channel("cluster, mid-high, L", 12, Pyr),
    Channel("cluster, mid-high, R", 13, SO),
    Channel("cluster, top, L", 14, SO),
    Channel("cluster, top, R", 15, SO),
)

_L2_channel_combinations = {
    "all": sorted(tuple(ch.index for ch in _L2_channels)),
    "pyr": (12,),
    "sr": (3,),
    "pyr+sr": (3, 12),
    "tetr": (10, 11, 12, 13),
}

CHANNEL_COMBINATIONS = _L2_channel_combinations
