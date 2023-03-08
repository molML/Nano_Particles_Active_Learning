"""

Derek van Tilborg | 08-03-2023  | Eindhoven University of Technology

"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def dbal(screen_df, m: int = 100, k: int = 10, seed: int = 42):

    # 1. select top m most uncertain samples
    # 2. perform k-means clustering on features of m, with k being the batch size
    # 3. select the k samples closest to their respective cluster centroid
    # 4. return k samples

    top_m = screen_df.nlargest(m, 'y_uncertainty')

    X_m = np.array(top_m.loc[:, ['x_PLGA', 'x_PP-L', 'x_PP-COOH', 'x_PP-NH2', 'x_S/AS']])
    id_m = np.array(top_m['ID'])

    kmeans = KMeans(n_clusters=k, random_state=seed).fit(X_m)

    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_m)

    return id_m[closest]
