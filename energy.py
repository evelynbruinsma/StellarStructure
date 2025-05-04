import constants as c
import numpy as np

def pp_rate(rho, T, X=0.7):
    '''
    calculate energy generation rate for pp chain using equation 18.63 in Kippenhahn et al. (2012)
    inputs:
        rho: float
            density (g/cm^3)
        T: float
            tempterature (K)
        X: float
            hydrogen mass fraction
    returns:
        pp_rate: float
            rate of pp chain (erg/g/s)
    '''
    T9 = T / (10**9)
    T7 = T / (10**7)
    psi = 1

    ED_kT = (5.92E-3) * (rho/T7**3)**(0.5)  # equation 18.56, assuming squiggle=1
    f11 = np.exp(ED_kT)  # equation 18.57, screening factor
    g11 = 1 + 3.82*T9 + 1.51*T9**2 + 0.144*T9**3 - 0.0114*T9**4

    pp_rate = (2.57E4) * psi * f11 * g11 * rho * X**2 * T9**(-2/3) * np.exp(-3.381/T9**(1/3))

    return pp_rate


def cno_rate(rho, T, X=0.7, Z=0.02):
    '''
    calculate energy generation rate for CNO cycle using equation 18.65 in Kippenhahn et al. (2012)
    inputs:
        rho: float
            density (g/cm^3)
        T: float
            tempterature (K)
        X: float
            hydrogen mass fraction
        Z: float
            CNO mass fraction (1-X-Y)
    returns:
        CNO_rate: float
            rate of CNO cycle (erg/g/s)
    '''
    T9 = T / (10 ** 9)

    g14_1 = 1 - 2.00*T9 + 3.41*T9**2 -2.43*T9**3
    cno_rate = (8.24E25) * g14_1 * Z * X * rho * T9**(-2/3) * np.exp(-15.231*T9**(-1/3) - (T9/0.8)**2)

    return cno_rate


def del_val(m, l, P, kappa, T, ad_v_rad = None):
    '''
    determine value for del (rad vs ad)
    inputs:
        m: float
            mass (g)
        l: float
            luminosity (erg/s)
        P: float
            pressure (dyne/cm^2)
        kappa: float
            opacity
        T: float
            temperature (K)
        ad_v_rad: str
            which value to return, return whichever is smallest if true
            default: none
    returns:
        del: float
            value of the gradient
    '''
    del_ad = 0.4  # adiabatic gradient
    del_rad = (3/(16*np.pi*c.a*c.c))*(P*kappa/T**4)*(l/(c.G*m))  # radiative gradient

    if ad_v_rad == None: # compare them, return min
        return np.min([del_ad, del_rad])
    elif ad_v_rad == 'ad':  # return adiabatic grad
        return del_ad
    elif ad_v_rad == 'rad':  # return radiative grad
        return del_rad