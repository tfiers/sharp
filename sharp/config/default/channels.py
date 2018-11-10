from dataclasses import dataclass
from typing import Optional

# `L2` is the L-style-probe recording at:
# >> nerffs01/ratlab/Frederic/organized/Long Flexible Probes/For the paper/data/P2/30_08_2014/Nlx/L2

# From: ./probe.svg
# Units: micrometre
L_probe_outline = (
    (159.3, 695.8),
    (129, 663.6),
    (9.5, 190.3),
    (9.5, 8.8),
    (159.3, 8.8),
)

x_left = 117.8
x_right = 147.7


SO = "Stratum oriens"
Pyr = "Pyramidal cell layer"
SR = "Stratum radiatum"
SLM = "Stratum lacunosum-moleculare"


@dataclass
class Channel:
    name: str
    index: int
    recording_site: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None


# The recording sites are inaccurate guesses, based on the Kjonigsen composition
# I made.
L2_channels = (
    Channel("linear-array: bottom", 0, SLM, x=x_right, y=663.5),
    Channel("linear-array: bottom+1", 1, SR, x=x_right, y=602.0),
    Channel("linear-array: bottom+2", 2, SR, x=x_right, y=539.5),
    Channel("linear-array: mid-low", 3, SR, x=x_right, y=478.1),
    Channel("linear-array: mid-high", 4, SR, x=x_right, y=419.0),
    Channel("linear-array: top-2", 5, SR, x=x_right, y=357.6),
    Channel("linear-array: top-1", 6, SR, x=x_right, y=295.1),
    Channel("linear-array: top", 7, Pyr, x=x_right, y=233.6),
    Channel("cluster: bottom, L", 8, Pyr, x=x_left, y=189.4),
    Channel("cluster: bottom, R", 9, Pyr, x=x_right, y=173.1),
    Channel("cluster: mid-low, L", 10, Pyr, x=x_left, y=159.1),
    Channel("cluster: mid-low, R", 11, Pyr, x=x_right, y=142.7),
    Channel("cluster: mid-high, L", 12, Pyr, x=x_left, y=128.3),
    Channel("cluster: mid-high, R", 13, SO, x=x_right, y=111.9),
    Channel("cluster: top, L", 14, SO, x=x_left, y=97.9),
    Channel("cluster: top, R", 15, SO, x=x_right, y=81.6),
)

# 0-based
L2_channel_combinations = {
    "all": sorted(tuple(ch.index for ch in L2_channels)),
    "sr": (3,),
    "pyr": (11,),
    "pyr+sr": (3, 11),
    # "sr-clust": (2, 3, 4, 5, 6),
    "sr-only-clust": (1, 2, 3, 4),
    "tetr": (10, 11, 12, 13),
}
