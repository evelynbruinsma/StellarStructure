# import necessary packages
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def opacity(T, rho):
    '''
    calculates the log of the rosseland mean opacity using interpolation
    parameters:
        file: str
            string of file name
        T: float
            temp (k)
        rho: float
            density (g/cm^3)
    returns:
        10**interpolation: flot
            estimated opacity value (not log)
    '''
    logT = np.log10(T)
    log_rho = np.log10(rho)
    logR = np.log10((10**log_rho) / ((10**logT)*(10**(-6)))**3)  #  log (density/T6)

    file = 'opacities2.txt'
    df = pd.read_csv(file, delim_whitespace=True)
    df = df.replace([9.999], np.nan)
    df_opacities = df.drop(columns=["logT"])  # df w/out temp column

    x = np.asarray(df['logT'])
    z = np.asarray(df_opacities)
    y = np.asarray(df_opacities.columns, dtype=float)

    interp = RegularGridInterpolator((x, y), z, bounds_error=False, fill_value=None)
    interpolation = interp((logT, logR))

    return(10**interpolation)
