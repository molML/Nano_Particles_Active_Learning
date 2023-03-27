"""
Code to generate the virtual screening library. We use a dirichlet distribution to generate formulation ratios.
Solvent/Anti-solvent ratios is a uniform discrete step distribution. All formulations are bound within sensible
experimental limits and are not allowed to overlap in experimental error.

Derek van Tilborg | 06-03-2023 | Eindhoven University of Technology

"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def generate_screen_data(size: int = 100000, seed: int = 42) -> np.ndarray:

    rng = np.random.default_rng(seed)
    size_extra = int(size * 1.2)

    experimental_error = {'PLGA': 1.2490, 'PP-L': 1.2121, 'PP-COOH': 1.2359, 'PP-NH2': 1.2398, 'S/AS': 100}  # TODO what do we do with S/AS?
    bounds = {k: (((100 - j) / 100), ((100 + j) / 100)) for k, j in experimental_error.items()}

    s_as = [0.10, 0.15, 0.20, 0.25, 0.30]
    exp_lims = {'PLGA': [0, 0.7],
                'PP-L': [0, 1],
                'PP-COOH': [0, 1],
                'PP-NH2': [0, 1],
                'S/AS': [min(s_as), max(s_as)]}

    # generate formulations. We use a dirichlet distribution to make sure all PLGA, PP-L, PP-COOH, and PP-NH2 ratios
    # add up to 1.
    x = rng.dirichlet(np.ones(4), size=size_extra)
    x_s_as = np.array([np.array(s_as)[rng.integers(0, len(s_as), size=size_extra)]]).T

    # Add S/AS column to the rest of the data
    x = np.append(x, x_s_as, axis=1)

    # filter formulations based on some sensible experimental limitations. Formulations should be within these limits
    x = x[np.where((x[:, 0] > exp_lims['PLGA'][0])      & (x[:, 0] < exp_lims['PLGA'][1]) &
                   (x[:, 1] > exp_lims['PP-L'][0])      & (x[:, 1] < exp_lims['PP-L'][1]) &
                   (x[:, 2] > exp_lims['PP-COOH'][0])   & (x[:, 2] < exp_lims['PP-COOH'][1]) &
                   (x[:, 3] > exp_lims['PP-NH2'][0])    & (x[:, 3] < exp_lims['PP-NH2'][1]))]

    # the error for every variable should not overlap
    within_error_range = []
    samples_found = []
    for i, row in tqdm(enumerate(x), 'Looking for formulations that overlap in error'):
        # Here I check if the error overlaps for ALL variables. The rationale behind this is that in the case some
        # variables are considered identical because of overlapping errors, it is still a different formulation if
        # the remaining variables are different.
        js = np.where((x[:, 0] > (row[0] * bounds['PLGA'][0]))     & (x[:,0] < (row[0] * bounds['PLGA'][1])) &
                      (x[:, 1] > (row[1] * bounds['PP-L'][0]))     & (x[:,1] < (row[1] * bounds['PP-L'][1])) &
                      (x[:, 2] > (row[2] * bounds['PP-COOH'][0]))  & (x[:,2] < (row[2] * bounds['PP-COOH'][1])) &
                      (x[:, 3] > (row[3] * bounds['PP-NH2'][0]))   & (x[:,3] < (row[3] * bounds['PP-NH2'][1])) &
                      (x[:, 4] > (row[4] * bounds['S/AS'][0]))     & (x[:,4] < (row[4] * bounds['S/AS'][1])))[0]

        if len(js) > 1:
            # remove self
            js = js[js != i]
            # check if the hits are not already picked up earlier. Only one of the overlapping pair should be removed.
            within_error_range += [j for j in list(js) if j not in samples_found]
            samples_found.append(i)

    print(f"Removing {len(within_error_range)} formulations that overlap in experimental error.")

    # remove the found overlapping samples from x
    x = x[[i for i in range(len(x)) if i not in within_error_range]]

    # Remove the excess
    x = x[range(size)]

    return x


if __name__ == '__main__':

    x = generate_screen_data(size=100000, seed=42)  # 1202 formulations were removed due to error overlaps

    df = pd.DataFrame(x, columns=['PLGA', 'PP-L', 'PP-COOH', 'PP-NH2', 'S/AS'])
    df['ID'] = [f"screen_{i}" for i in range(len(df))]

    df.to_csv('data/screen_library.csv', index=False)
