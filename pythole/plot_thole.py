import itertools
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn

from pythole.databases import _OUT_DATA_DIR

_PREDICTIONS = _OUT_DATA_DIR / "predictions"
_IMGS = _OUT_DATA_DIR / "imgs"
alpha_kinds = ("atomtype", "free", "mbis")
embed_kinds = ("cm5", "hir", "mbis")

overall_max = 50
overall_min = -150
j = 0
i = 0
axes = None
show = True
fname = ""
for alpha_kind, embed_kind in itertools.product(alpha_kinds, embed_kinds):
    title = f"Polarizability: {alpha_kind}, Embedding: {embed_kind}"

    for f in sorted(_PREDICTIONS.rglob(f"*{alpha_kind}.npy")):
        if f.stem.startswith("embed"):
            continue
        _, _damp, _epsilon, _ = f.stem.split("-")
        damp = float(_damp)
        epsilon = float(_epsilon.replace("9999", "inf")) / 10
        i += 1
        if j == 0:
            if axes is not None:
                if show:
                    plt.show()
                if damp == 0.0:
                    plt.savefig(_IMGS / f"{fname}-highthole.png")
                else:
                    plt.savefig(_IMGS / f"{fname}-lowthole.png")
            fig, axes = plt.subplots(
                4,
                5,
                sharex=True,
                sharey=True,
                layout="compressed",
                dpi=142,
                figsize=(13.5, 7.6),
            )
            fig.suptitle(title)
            axes = axes.reshape(-1)

        assert axes is not None
        ax = axes[j]
        correction = np.load(f)
        if (correction == 0).all():
            correction = 0.0
        pred = correction + np.load(_PREDICTIONS / f"embed-{embed_kind}.npy")

        #  Filter outliers
        target = np.load(_PREDICTIONS / "target.npy")
        outliers = np.abs(pred - target) > 200
        pred = np.delete(pred, outliers)
        target = np.delete(target, outliers)

        # Calculate summary statistics
        pearsonr_squared = scipy.stats.pearsonr(pred, target).statistic ** 2
        err = pred - target
        mad = np.abs(err).mean()
        mrad = np.abs(err / target).mean()
        rmse = np.sqrt((err**2).mean())
        slope, _, _, _ = np.linalg.lstsq(target[:, None], pred, rcond=None)

        # Plot diagonal lines
        ax.plot(target, target * slope, linestyle="solid", color="black", linewidth=1.0)
        ax.plot(target, target, linestyle="solid", color="silver", linewidth=0.5)

        # Plot scatter and hist
        seaborn.scatterplot(x=target, y=pred, s=5, color=".15", ax=ax)
        seaborn.histplot(x=target, y=pred, bins=300, pthresh=0.001, cmap="mako", ax=ax)

        # Plot summary statistics table
        string = [
            r"Pearson's $R^2$: " f"{pearsonr_squared:.3f}",
            f"MAD: {mad:.3f} kcal/mol",
            f"RMSE: {rmse:.3f} kcal/mol",
            f"MRAD: {mrad:.4f}",
            r"$\epsilon$: " f"{epsilon:.3f}",
            r"$a_{Thole}$: " f"{damp:.3f}",
        ]
        ax.text(
            0.01, 0.99, s="\n".join(string), transform=ax.transAxes, ha="left", va="top"
        )

        ax.set_xlim(overall_min, overall_max)
        ax.set_ylim(overall_min, overall_max)
        ax.set_xlabel(r"Target $E_{embed} + E_{pol}$, kcal/mol")
        ax.set_ylabel(r"Pred. $E_{embed} + E_{pol}$, kcal/mol")
        j = (j + 1) % 20
        fname = f"{alpha_kind}-{embed_kind}"

if show:
    plt.show()
plt.savefig(_IMGS / f"{fname}-highthole.png")
