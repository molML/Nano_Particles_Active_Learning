
import numpy as np
import pandas as pd
from copy import deepcopy


def scatter(y, y_hat, uncertainty=None, labels=None, lower=None, upper=None, outfile: str = None):

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
    plt.figure(figsize=(20, 20))
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))

    if outfile is not None:
        fig.savefig(outfile, dpi=150)

    plt.show()


def picks_pca(df, screen_x, picks):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import pandas as pd

    df = deepcopy(df)
    df['pick'] = 0
    df.loc[df['ID'].isin(picks), 'pick'] = 1
    df['alph'] = 0.01
    df.loc[df['ID'].isin(picks), 'alph'] = 1

    pca_screen = PCA(n_components=2)
    pca_screen = pca_screen.fit_transform(screen_x)
    pca_df = pd.DataFrame(data=pca_screen, columns=['PC 1', 'PC 2'])

    df['PC1'] = pca_df['PC 1']
    df['PC2'] = pca_df['PC 2']
    df.to_csv('results/pca.csv')

    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x="PC 1", y="PC 2",
        hue=df['pick'],
        data=pca_df,
        alpha=df['alph']
    )
    plt.show()
