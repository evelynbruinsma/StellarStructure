# import necessary packages
import matplotlib.pyplot as plt
import numpy as np
import scipy

# import necessary python scripts
import constants as c
import density
import energy
import IBC
import make_plots
import opacity

def residuals(scales, initial_guess, args):
    '''
    determines residuals for best fit optimization
    inputs:
        scales: list
            scale factors used on initial guess
        initial_guess: list
            list of initial parameter values [L, P, R, T]
        args: list
            arguments to pass to scipy.optimize.minimize in optimize
    returns:
        np.sum(dif2): float
    '''
    vars = initial_guess * scales
    sol_center, sol_surface = IBC.shootf(vars, args)
    dif = (sol_center.y[:, -1] - sol_surface.y[:, -1]) / (sol_center.y[:, 0] - sol_surface.y[:, 0])
    dif2 = np.sum(dif**2)
    return np.sum(dif2)


def optimize(guess):
    '''
    optimizes initial guess values to provide best-fit parameters
    inputs:
        guess: list
            list of initial guess values [L, P, R, T]
    returns:
        best_fit: list
            list of optimized best-fit values
    '''
    scales = [1, 1, 1, 1]
    bounds = np.array([[0.5] * 4, [1.5] * 4]).T
    fit = scipy.optimize.minimize(residuals, x0=scales, args=(guess, args), bounds=bounds, method='L-BFGS-B')
    print('Number of iterations: {}'.format(fit.nit))
    best_fit = initial_guess * fit.x

    return best_fit


M = 1.2 * c.Ms  # mass of star
X = 0.7  # hydrogen mass fraction
Y = 0.28  # helium mass fraction
Z = 1 - X - Y  # metals mass fraction

# initial guesses
R = 1.50 * c.Rs * (M / c.Ms)**0.75  # radius (cm), homology relations from class
L = 1.25 * c.Ls * (M / c.Ms)**3.5  # luminosity (erg/s), homology relations from class
Pc = (1.92e17)*1.42  # central pressure (dyne/cm^2), scaled solar value
Tc = (1.57e7)*1.42  # central temp (K), scaled solar value

initial_guess = np.array([L, Pc, R, Tc])
args = [M]

# solution to initial guess
sol_cen, sol_surf = IBC.shootf(initial_guess, args, M_start=1e-5)  # solutions to initial guess

# plots of initial guesses
make_plots.plot_separately(sol_cen, sol_surf, 1)
make_plots.plot_together(sol_cen, sol_surf, M, 1)

# plots of best fit initial guesses
best_fit = optimize(initial_guess)
sol_cen_bf, sol_surf_bf = IBC.shootf(best_fit, args, M_start=1e-5)
make_plots.plot_separately(sol_cen_bf, sol_surf_bf, 2)
make_plots.plot_together(sol_cen_bf, sol_surf_bf, M, 2)

# compare best fit results with MESA
make_plots.plot_separately(sol_cen_bf, sol_surf_bf, 3, mesa = True)

# save stellar parameters in a csv
make_plots.save_csv(sol_cen_bf, sol_surf_bf)

# calculate percent difference between model and MESA
make_plots.percent_dif(sol_cen_bf, sol_surf_bf)




