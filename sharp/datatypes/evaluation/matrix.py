from dataclasses import dataclass
import numpy as np
from fileflow import Saveable, File


@dataclass
class PerformanceMatrix(Saveable):
    
    ORF_performance: np.ndarray
    RNN_performance: np.ndarray
    
    def get_filetype():
        return PerformanceMatrixFile



class PerformanceMatrixFile(File):
    ...