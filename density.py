import constants as c
import numpy as np

def density(P, T, X=0.7, Y=0.28):
    '''
    calculates density and radiation pressure/total pressure ratio
    inputs:
        P: float
            pressure (dyne/cm^2)
        T: float
            temperature (K)
        X: float
            H mass fraction
        Y: float
            He mass fraction
    returns:
        rho: float
            density (g/cm^3)
    '''
    mu = 4 / (6*X + Y + 2)  # mean molecular weight
    rho = (P*mu*c.mp) / (c.k*T)  # density

    return rho
