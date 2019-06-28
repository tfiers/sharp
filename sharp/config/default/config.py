from sharp.config.default.raw_data import fklab_data
from sharp.config.spec import SharpConfig


config = SharpConfig(output_root="output/", raw_data=fklab_data, fs_target=1000)
