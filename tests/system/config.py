from pathlib import Path

from sharp.config.config import SharpConfig


config = SharpConfig(
    output_root="data/processed/",
    raw_data={
        "rat_2__day_5__probe_D29": Path("data/raw/sig.moz"),
        "rat_2__day_2__probe_L2": Path("data/raw/sig.dat"),
        "rat_3__day_1__probe_Waluigi": Path("data/raw/sig.raw.kwd"),
    },
    epochs=3,
)
