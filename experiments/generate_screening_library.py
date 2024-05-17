"""
Code to generate the virtual screening library. We use a dirichlet distribution to generate formulation ratios.
Solvent/Anti-solvent ratios is a uniform discrete step distribution. All formulations are bound within sensible
experimental limits and are not allowed to overlap in experimental error.

Derek van Tilborg | 06-03-2023 | Eindhoven University of Technology

"""
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm


def generate_screen_data(size: int = 100000, seed: int = 42) -> np.ndarray:

    rng = np.random.default_rng(seed)
    size_extra = int(size * 1.5)  # we make some extra to account for the ones that will be filtered out

    experimental_error = {'PLGA': 1.2490, 'PP-L': 1.2121, 'PP-COOH': 1.2359, 'PP-NH2': 1.2398, 'S/AS': 100}
    bounds = {k: (((100 - j) / 100), ((100 + j) / 100)) for k, j in experimental_error.items()}

    s_as = [0.10, 0.15, 0.20, 0.25]
    exp_lims = {'PLGA': [0, 0.7],
                'PP-L': [0, 1],
                'PP-COOH': [0, 1],
                'PP-NH2': [0, 1],
                'S/AS': [min(s_as), max(s_as)]}

    # generate formulations. We use a dirichlet distribution to make sure all PLGA, PP-L, PP-COOH, and PP-NH2 ratios
    # add up to 1.
    x = rng.dirichlet(np.ones(4), size=size_extra)

    # We check for samples that are lower than 0.06 and turn them into 0 (because of the pump's carryover volume)
    # The residuals from the samples < 0.06 will get moved to a random other variable (that is not 0).
    # Looping over the data multiple times is a bit of a duct-tape engineering solution, but only takes a few seconds.
    for _ in range(4):
        for formulation in x:
            for i, value in enumerate(formulation):
                if value < 0.06:
                    other_idxs = [j for j in range(4) if j != i]  # get the remaining indices
                    other_idxs = [j for j in other_idxs if formulation[j] != 0]  # make sure we're not selecting 0s
                    if len(other_idxs) > 0:  # if there are any, make the chosen index 0 and add its value elsewhere
                        formulation[i] = 0
                        random_idx_to_add_to = other_idxs[rng.integers(0, len(other_idxs), 1)[0]]
                        formulation[random_idx_to_add_to] += value

    # Add S/AS column to the rest of the data
    x_s_as = np.array([np.array(s_as)[rng.integers(0, len(s_as), size=size_extra)]]).T
    x = np.append(x, x_s_as, axis=1)

    # filter formulations based on some sensible experimental limitations. Formulations should be within these limits
    x = x[np.where((x[:, 0] >= exp_lims['PLGA'][0])      & (x[:, 0] < exp_lims['PLGA'][1]) &
                   (x[:, 1] >= exp_lims['PP-L'][0])      & (x[:, 1] < exp_lims['PP-L'][1]) &
                   (x[:, 2] >= exp_lims['PP-COOH'][0])   & (x[:, 2] < exp_lims['PP-COOH'][1]) &
                   (x[:, 3] >= exp_lims['PP-NH2'][0])    & (x[:, 3] < exp_lims['PP-NH2'][1]))]

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

    # Remove the excess if there is any
    if len(x) > size:
        x = x[range(size)]
    else:
        warnings.warn(f'Could not generate {size} formulations, generated {len(x)} instead')

    return x


def estimate_design_space_combinations() -> int:
    """ Estimate the numnber unique formulations. We first calculate the number of distinguishable points for each
    variable and then check if they sum up to one.

    :return: number of possible unique nano particle combinations within the design space
    """

    # Define the range and error of every formulation variable
    x1_lower, x1_upper, e1 = 0.06, 0.7, 0.012490  # PLGA
    x2_lower, x2_upper, e2 = 0.06, 1, 0.012121  # PP-L
    x3_lower, x3_upper, e3 = 0.06, 1, 0.012121  # PP-COOH
    x4_lower, x4_upper, e4 = 0.06, 1, 0.012398  # PP-NH2
    e_mean = np.mean([e1, e2, e3, e4])

    def discretize_variable(lower_bound: float, upper_bound: float, error: float):
        x = [lower_bound]
        while x[-1] <= upper_bound * (1 - error):
            x.append(x[-1] * (1 + error))
        return np.array(x)

    x1 = discretize_variable(x1_lower, x1_upper, e1)
    x2 = discretize_variable(x2_lower, x2_upper, e2)
    x3 = discretize_variable(x3_lower, x3_upper, e3)
    x4 = discretize_variable(x4_lower, x4_upper, e4)

    # Loop over all combinations and check if they sum to one
    N = 0
    for x1_ in tqdm(x1):
        for x2_ in x2:
            for x3_ in x3:
                for x4_ in x4:
                    if (1 - e_mean) < sum([x1_, x2_, x3_, x4_]) < (1 + e_mean):
                        N += 1

    # Multiply the possible combinations by the 4 possible solvent/antisolvent ratios
    N = N * 4

    return N


if __name__ == '__main__':

    x = generate_screen_data(size=100000, seed=42)

    df = pd.DataFrame(x, columns=['PLGA', 'PP-L', 'PP-COOH', 'PP-NH2', 'S/AS'])
    df['ID'] = [f"screen_{i}" for i in range(len(df))]

    df.to_csv('data/screen_library.csv', index=False)

    print("Estimated total design space:", estimate_design_space_combinations())
