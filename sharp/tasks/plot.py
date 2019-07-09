from typing import Sequence

from matplotlib.figure import Figure

from sharp.datatypes.evaluation import PerformanceMatrix, ThresholdSweep


def plot_PR_curve(sweep: ThresholdSweep) -> Figure:
    ...


def plot_F_matrix(matrix: PerformanceMatrix) -> Figure:
    ...


def plot_multirec_F_matrix(matrices: Sequence[PerformanceMatrix]) -> Figure:
    ...
