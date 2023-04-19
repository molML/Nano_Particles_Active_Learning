"""

Acquisition function for explorative and exploitative sample selection.

Explorative acquisition is based on the "Diverse mini-Batch Active Learning" algorithm [1]. Here samples are first
filtered by their uncertainty (m), after which the k-most representative samples are selected from this subset by kmeans
clustering. Exploitative acquisition is a simple two-prong subset algorithm where first the m highest predicted samples
are selected, after which we take the k-most certain ones from this subset. In both cases we first remove any samples
that are likely very unstable (predicted PdI cutoff of 0.2)

[1] Zhdanov, F. (2019). Diverse mini-batch active learning. arXiv preprint arXiv:1901.05954.

Derek van Tilborg | 13-03-2023  | Eindhoven University of Technology

"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def acquisition_function(screen_df, m: int = 100, k: int = 10, seed: int = 42, mode: str = 'explorative',
                         pdi_cutoff: float = 0.2, size_cutoff: float = None, previous_picks=None):
    """ Acquisition algorithm
    Explorative:
        1. select top m most uncertain samples
        2. perform k-means clustering on features of m, with k being the batch size
        3. select the k samples closest to their respective cluster centroid
        4. return k samples
    Exploitation:
        1. select top m highest predictions
        2. select top k most certain samples, with k being the batch size
        3. return k samples

    :param screen_df: Pandas dataframe with all predictions. Should contain the following columns:
        ['ID', 'y_hat_uptake', 'y_uncertainty_uptake', 'y_hat_pdi', 'y_uncertainty_pdi', 'x_PLGA', 'x_PP-L',
        'x_PP-COOH', 'x_PP-NH2', 'x_S/AS']
    :param m: top m subset size
    :param k: batch size
    :param seed: random seed
    :param mode: 'explorative' or 'exploitative'
    :param pdi_cutoff: cutoff to remove likely unstable formulations
    :return: nd.array with sample ids
    """

    assert mode in ['explorative', 'exploitative'], f"'mode' should be 'explorative' or 'exploitative'. Not {mode}"

    # Remove all formulations that have already been picked in previous cycles
    if previous_picks is not None:
        screen_df = screen_df.loc[~screen_df['ID'].isin(previous_picks)]

    # Remove all formulations with a predicted PdI higher than the cuttoff value
    if pdi_cutoff is not None:
        screen_df = screen_df.loc[screen_df['y_hat_pdi'] < pdi_cutoff]

    # Remove all formulations with a predicted size higher than the cuttoff value
    if size_cutoff is not None:
        screen_df = screen_df.loc[screen_df['y_hat_size'] < size_cutoff]

    if mode == 'explorative':
        # select top m most uncertain samples
        top_m = screen_df.nlargest(m, 'y_uncertainty_uptake')

        # perform k-means clustering on features of m, with k being the batch size
        X_m = np.array(top_m.loc[:, ['x_PLGA', 'x_PP-L', 'x_PP-COOH', 'x_PP-NH2', 'x_S/AS']])
        id_m = np.array(top_m['ID'])
        kmeans = KMeans(n_clusters=k, random_state=seed).fit(X_m)

        # select the k samples closest to their respective cluster centroid
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_m)

        # return k samples
        return id_m[closest]

    if mode == 'exploitative':
        # select top m highest predicted samples
        top_m = screen_df.nlargest(m, 'y_hat_uptake')
        # select top k most **certain** samples
        top_k = top_m.nsmallest(k, 'y_uncertainty_uptake')

        return np.array(top_k['ID'])
