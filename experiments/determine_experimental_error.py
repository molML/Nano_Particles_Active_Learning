import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import math


def get_pipetting_error(volume: float) -> float:
    """ Calculate the pipetting error of 200ul/1000ul eppendorf pipettes. Input volume is in ul. output error in %
    https://www.eppendorf.com/product-media/doc/en/672922/Liquid-Handling_Technical-data_Research-plus_Eppendorf-Research-plus.pdf

    """
    assert volume <= 1000, 'Pipetting volume too large. Only 10-1000 ul is known'

    pipetting_errors = pd.DataFrame({'volume_(ul)':       [10, 20, 25, 50, 100, 200, 1000],
                                    'systematic_error_%': [1.2, 1, 1, 0.7, 0.6, 0.6, 0.6],
                                    'random_error_%':     [0.6, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2]})

    # get the errors corresponding to the pipetting volume.
    row = pipetting_errors[pipetting_errors['volume_(ul)'] < volume][-1:]

    # since the systematic error and the random error are independent, we take the quadrature of the individual errors
    total_pipetting_error = math.sqrt(row['systematic_error_%']**2 + row['random_error_%']**2)

    return total_pipetting_error


def get_balance_error(mass: float) -> float:
    """ Get the percentage error of the balance for a given mass (in mg). Return error in percentage """
    balance_error = 0.01  # mg

    return balance_error/mass*100


def get_stock_solution_error(mass: float, volume: float) -> float:
    """ mass in mg, volume in ul """

    return math.sqrt(get_pipetting_error(volume) ** 2 + get_balance_error(mass) ** 2)


if __name__ == '__main__':

    """In our case, experimental errors are additive. We simply take the quadrature of all compounding errors.
    Firstly, we have a pipetting error consisting of a systematic error and a random error, depending on the pipetting 
    volume. Secondly, the balance used to weigh all compounds has an error, and finally, the pump has an error. We 
    ignore the carryover volume of the pump, as it only applies to the last used stock solution. """

    PUMP_ERROR = 1  # We assume 1% pump error (which is the most extreme scenario)

    # PLGA
    plga_error = get_stock_solution_error(mass=2.5, volume=167)  # pipetting + balance error
    plga_error = math.sqrt(plga_error ** 2 + PUMP_ERROR ** 2)    # include pump error

    # PP-L
    ppl_error = get_stock_solution_error(mass=3.8, volume=253)  # pipetting + balance error
    ppl_error = math.sqrt(ppl_error ** 2 + PUMP_ERROR ** 2)     # include pump error

    # PP-COOH
    ppcooh_error = get_stock_solution_error(mass=2.8, volume=187)   # pipetting + balance error
    ppcooh_error = math.sqrt(ppcooh_error ** 2 + PUMP_ERROR ** 2)   # include pump error

    # PP-NH2
    ppnh2_error = get_stock_solution_error(mass=2.7, volume=180)  # pipetting + balance error
    ppnh2_error = math.sqrt(ppnh2_error ** 2 + PUMP_ERROR ** 2)   # include pump error

    print(f"PLGA error (%)    \t {plga_error:.4f}\n"
          f"PP-L error (%)    \t {ppl_error:.4f}\n"
          f"PP-COOH error (%) \t {ppcooh_error:.4f}\n"
          f"PP-NH2 error (%)  \t {ppnh2_error:.4f}")

    # PLGA error (%)    	 1.2490
    # PP-L error (%)    	 1.2121
    # PP-COOH error (%) 	 1.2359
    # PP-NH2 error (%)  	 1.2398

    # The total experimental error in %
    experimental_error = {'PLGA': 1.2490, 'PP-L': 1.2121, 'PP-COOH': 1.2359,  'PP-NH2': 1.2398,  'S/AS': 0}
