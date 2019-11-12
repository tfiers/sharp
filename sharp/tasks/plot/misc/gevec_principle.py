from matplotlib.text import Text
from numpy import array, real_if_close, sqrt
from numpy.linalg import svd
from numpy.random import normal
from pandas import DataFrame, concat
from scipy.linalg import eigh

from seaborn import stripplot
from sharp.data.files.figure import FigureTarget
from sharp.data.types.aliases import subplots
from sharp.data.hardcoded.style import blue, green, orange, red
from sharp.tasks.base import SharpTask
from sharp.tasks.plot.base import FigureMaker

# Signal and noise covariance matrices:
Rss = array([[1.5, 1], [1, 3]])
Rnn = array([[4, 1], [1, 1]])

# Dimensions and number of data points:
M = 2
Num = 100

# Generate toy data drawn from a multivariate normal distribution:
S = Rss @ normal(size=(M, Num))
N = Rnn @ normal(size=(M, Num))

# Singular value decompositions of data matrices:
SVecs, SVals, _ = svd(S)
SVecs_N, SVals_N, _ = svd(N)

# Empirical signal and noise covariance matrices:
RRss = S @ S.T
RRnn = N @ N.T

# Generalised eigenvalue decomposition of empirical signal and noise
# covariance matrices:
GEVals, GEVecs = eigh(RRss, RRnn)
GEVals = real_if_close(GEVals)


output_dir = FigureMaker.output_dir / "GEVec-principle"

colors = {"GEVec": green, "PCA": red, "Signal": blue, "Noise": orange}
fontsize = 22


class PlotGEVecPrinciple(SharpTask):
    def requires(self):
        return (ScatterPlot(), StripPlot())


class ScatterPlot(FigureMaker):
    def output(self):
        return FigureTarget(output_dir, "scatter")

    def work(self):
        fig, ax = subplots(figsize=(5, 5))
        scatter(S, ax, colors["Signal"])
        scatter(N, ax, colors["Noise"])
        plot_vector(SVals[0] * -SVecs[:, 0], ax, colors["PCA"])
        # plot_vector(SVals_N[1] * -SVecs_N[:,1], ax, 'black')
        # plot_vector(SVals[1] * SVecs[:,1], ax)
        plot_vector(40 * sqrt(GEVals[-1]) * GEVecs[:, -1], ax, colors["GEVec"])
        # plot_vector(40*np.sqrt(gevl[-2]) * gevc[:,-2], ax, 'C5')
        ax.set_aspect("equal")
        lims = 11
        ax.set_xlim(-lims, lims)
        ax.set_ylim(-lims, lims)
        for s, loc in (["Signal", (0.52, 0.26)], ["Noise", (0.7, 0.46)]):
            fig.text(*loc, s, color=colors[s], fontsize=fontsize)
        ax.set_xlabel("Electrode 3")
        ax.set_ylabel("Electrode 12")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        fig.tight_layout()
        self.output().write(fig)


def scatter(X, ax, c):
    ax.plot(X[0, :], X[1, :], ".", color=c)


def plot_vector(x, ax, c):
    ax.plot([0, x[0]], [0, x[1]], "-", color=c)


class StripPlot(FigureMaker):
    def output(self):
        return FigureTarget(output_dir, "strips")

    def work(self):
        # fmt: off
        df: DataFrame = concat((
                 DataFrame(dict(proj=S.T @ -SVecs[:, 0],
                                method='PCA',
                                src='Signal')),
                 DataFrame(dict(proj=N.T @ -SVecs[:, 0],
                                method='PCA',
                                src='Noise')),
                 DataFrame(dict(proj=S.T @ GEVecs[:, -1],
                                method='GEVec',
                                src='Signal')),
                 DataFrame(dict(proj=N.T @ GEVecs[:, -1],
                                method='GEVec',
                                src='Noise')),
        ))
        # fmt: on
        def normalise(method):
            select = (df.method == method, "proj")
            scale = max(abs(df.loc[select]))
            df.loc[select] /= scale

        normalise("PCA")
        normalise("GEVec")

        fig, ax = subplots()
        stripplot(data=df, x="proj", y="method", hue="src", dodge=0.5, ax=ax)
        ax.legend_.remove()
        ax.spines["bottom"].set_visible(True)
        ax.set_xticks([])
        ax.set_xlabel("Projection on 1st eigenvector")
        ax.set_ylabel("")
        ax.yaxis.set_tick_params(labelcolor="black", labelsize=fontsize, pad=16)
        for label in ax.get_yticklabels():
            label: Text
            label.set_color(colors[label.get_text()])
        fig.tight_layout()
        self.output().write(fig)
