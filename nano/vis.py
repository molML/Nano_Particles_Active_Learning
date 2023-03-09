
import numpy as np


def scatter(y, y_hat, uncertainty=None, labels=None):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(y, y_hat)
    ax.set_xlim(min(np.append(y, y_hat)), max(np.append(y, y_hat)))
    ax.set_ylim(min(np.append(y, y_hat)), max(np.append(y, y_hat)))
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", alpha=0.5)
    if uncertainty is not None:
        plt.errorbar(y, y_hat, yerr=uncertainty, fmt="o", alpha=0.5)
    if labels is not None:
        for i, txt in enumerate(labels):
            ax.annotate(txt, (y[i], y_hat[i]))
    plt.xlabel("True")
    plt.ylabel("Predicted")
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))
    ax.grid(b=True, which='major', color='k', linestyle='--', alpha=0.25)

    plt.show()
