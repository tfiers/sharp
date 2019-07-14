from typing import Sequence

from matplotlib.figure import Figure

from sharp.datatypes.evaluation.sweep import ThresholdSweep
from sharp.datatypes.evaluation.matrix import PerformanceMatrix
from sharp.init import sharp_workflow


@sharp_workflow.task
def plot_PR_curve(sweep: ThresholdSweep) -> Figure:
    ...


@sharp_workflow.task
def plot_F_matrix(matrix: PerformanceMatrix) -> Figure:
    ...


@sharp_workflow.task
def plot_multirec_F_matrix(matrices: Sequence[PerformanceMatrix]) -> Figure:
    ...
