from preludio.preload import preload_with_feedback


preload_with_feedback(["scipy.signal", "fklab.segments", "matplotlib.pyplot"])

from sharp.config import SharpConfig

config = SharpConfig.load()
