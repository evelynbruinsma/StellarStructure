# import necessary packages
import constants as c
import numpy as np
import scipy

# import necessary python scripts
import density
import energy
import opacity

def load_center(Pc, Tc):
    '''
    integrates from the center out to find a solution
    inputs:
        Pc: float
            central pressure (dyne/cm^2)
        Tc: float
            central temperature (K)
    returns:
        array of [luminosity (erg/s), pressure (dyne/cm^2), radius (cm), temperature (K)]
    '''
    rho_c = density.density(P=Pc, T=Tc)
    m = (1e-5) * c.Ms  # mass point just outside core center
    epsilon = energy.pp_rate(rho_c, Tc) + energy.cno_rate(rho_c, Tc)  # energy generation from pp and CNO
    l_core = epsilon * m  # core luminosity
    r = (3*m / (4*np.pi*rho_c))**(1/3)  # Kippenhahn, eqn. 11.3
    P = Pc - (3*c.G / (8*np.pi)) * (4*np.pi*rho_c/3)**(4/3) * m**(2/3)  # Kippenhahn, eqn. 11.6
    del_rad = energy.del_val(m, l_core, P, rho_c, Tc, ad_v_rad='rad')
    del_ad = energy.del_val(m, l_core, P, rho_c, Tc, ad_v_rad='ad')

    if del_rad > del_ad:  # --> convective core, Kippenhahn eqn. 11.9.2
        T = np.exp(np.log(Tc) - (np.pi/6)**(1/3) * c.G * (del_ad * rho_c**(4/3)*m**(2/3))/Pc)
    else:  # --> radiative core, Kippenhahn eqn. 11.9.1
        kappa_c = opacity.opacity(np.log10(Tc), np.log10(rho_c))
        T = (Tc**4 - (1 / (2*c.a*c.c) * (3 / (4*np.pi))**(2/3) * kappa_c * epsilon * rho_c**(4/3) * m**(2/3)))**(1/4)

    # return guesses
    return np.array([l_core, P, r, T])


def load_surface(M, L, R):
    '''
    integrates from the surface in to find a solution
    inputs:
        M: float
            mass (g)
        L: float
            luminosity (erg/s)
        R: float
            radius (cm)
    returns:
        array of [luminosity (erg/s), pressure (dyne/cm^2), radius (cm), effective temperature (K)]
    '''
    kappa = 0.34  # typical value in solar type photosphere
    T_eff = (L / (4*np.pi*R**2*c.sb))**(1/4)  # effective temp (K)
    P = (2/3) * (c.G*M/R**2) * (1/kappa)  # at tau=2/3

    # return guesses
    return np.array([L, P, R, T_eff])


def derivs(m, y, args):
    '''
    calculate values for 4 coupled Diff Eqs that describe the stellar interior
    inputs:
        m: float
            mass (g)
        y: list
            values for luminosity (erg/s), pressure (dyne/cm^2), radius (cm), effective temperature (K)
        args: list
            arguments passed to solve_ivp in shootf
    returns:
        array of [dldm, dPdm, drdm, dTdm]
            dldm: deriv of luminosity WRT mass
            dPdm: deriv of pressure WRT mass
            drdm: deriv of radius WRT mass
            dTdm: deriv of temp WRT mass
    '''
    l, P, r, T = y
    rho = density.density(P=P, T=T)
    kappa = opacity.opacity(T, rho)
    del_ = energy.del_val(m, l, P, kappa, T, ad_v_rad=None)
    dldm = energy.pp_rate(rho, T) + energy.cno_rate(rho, T)  # energy generation from pp and CNO, dldm = epsilon
    dTdm = -(c.G*m*T*del_) / (4*np.pi*P*r**4)
    dPdm = - (c.G*m) / (4*np.pi*r**4)
    drdm = 1 / (4*np.pi*r**2*rho)

    return np.array([dldm, dPdm, drdm, dTdm])


def shootf(vals, args, M_mid = 0.5, M_start=1e-5):
    '''
    integrates until the inner and outer solutions meet at a midpoint
    inputs:
        vals: list
            luminosity (erg/s), pressure (dyne/cm^2), radius (cm), temperature (K)
        args: list
            arguments passed to solve_ivp
        M_mid: float
            midpoint mass value where solutions meet
        M_start: float
            starting mass value
    returns:
        [sol_center, sol_surface]: list
            inner and outer solutions
    '''
    L, P, R, T = vals
    M = args[0]
    steps = 10000
    M_center = np.linspace(M_start, M_mid, steps) * M
    sol_center = scipy.integrate.solve_ivp(derivs, (M_center[0], M_center[-1]), load_center(P, T), args=args, method='RK45', t_eval=M_center)
    M_surface = np.linspace(1, M_mid, steps) * M
    sol_surface = scipy.integrate.solve_ivp(derivs, (M_surface[0], M_surface[-1]), load_surface(M, L, R), args=args, method='RK45', t_eval=M_surface)

    return sol_center, sol_surface

