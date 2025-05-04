# import necessary packages
import matplotlib.pyplot as plt
import mesa_reader as mr
import numpy as np
import pandas as pd

# import necessary python scripts
import constants as c
import density
import energy
import opacity

def plot_separately(sol_cen, sol_surf, iteration, mesa=False):
    '''
    plot stellar parameters separately as a function of solar mass
    inputs:
        sol_cen: array
            solution from outward integrations (center out)
        sol_surf: array
            solution from inward integrations (surface in)
        iteration: int
            1, 2, or 3 (intial guess vs best-fit vs MESA comparison)
        mesa: bool
            determines whether to plot mesa values or not
            default: False (do not plot MESA values)
    returns:
        none, generates/saves plots
    '''
    if mesa == True:
        L_mesa, p = mesa_vals()

    # luminosity
    fig, ax = plt.subplots(2, 2, figsize=(7.5, 7))
    ax[0, 0].plot(sol_cen.t / c.Ms, sol_cen.y[0, :], label='Center', color='C2')
    ax[0, 0].plot(sol_surf.t / c.Ms, sol_surf.y[0, :], label='Surface', color='C4')
    if mesa == True:
        ax[0, 0].plot(p.mass, L_mesa, label='MESA', color='C6')
    ax[0, 0].set_title('Luminosity')
    ax[0, 0].set_xlabel('M/M$_\odot$')
    ax[0, 0].set_ylabel('L [erg/s]')
    ax[0, 0].set_yscale('log')
    ax[0, 0].legend()

    # pressure
    ax[0, 1].plot(sol_cen.t / c.Ms, sol_cen.y[1, :], label='Center', color='C2')
    ax[0, 1].plot(sol_surf.t / c.Ms, sol_surf.y[1, :], label='Surface', color='C4')
    if mesa == True:
        ax[0, 1].plot(p.mass, p.P, label='MESA', color='C6')
    ax[0, 1].set_title('Pressure')
    ax[0, 1].set_xlabel('M/M$_\odot$')
    ax[0, 1].set_ylabel('P [dyne/cm$^{2}$]')
    ax[0, 1].set_yscale('log')
    ax[0, 1].legend()

    # radius
    ax[1, 0].plot(sol_cen.t / c.Ms, sol_cen.y[2, :], label='Center', color='C2')
    ax[1, 0].plot(sol_surf.t / c.Ms, sol_surf.y[2, :], label='Surface', color='C4')
    if mesa == True:
        ax[1, 0].plot(p.mass, p.R*c.Rs, label='MESA', color='C6')
    ax[1, 0].set_title('Radius')
    ax[1, 0].set_xlabel('M/M$_\odot$')
    ax[1, 0].set_ylabel('R [cm]')
    ax[1, 0].set_yscale('log')
    ax[1, 0].legend()

    # temperature
    ax[1, 1].plot(sol_cen.t / c.Ms, sol_cen.y[3, :], label='Center', color='C2')
    ax[1, 1].plot(sol_surf.t / c.Ms, sol_surf.y[3, :], label='Surface', color='C4')
    if mesa == True:
        ax[1, 1].plot(p.mass, p.T, label='MESA', color='C6')
    ax[1, 1].set_title('Temperature')
    ax[1, 1].set_xlabel('M/M$_\odot$')
    ax[1, 1].set_ylabel('T [K]')
    ax[1, 1].set_yscale('log')
    ax[1, 1].legend()

    if iteration == 1:
        fig.suptitle('M=1.2 M$_\odot$: initial guess', fontsize=16)
    elif iteration == 2:
        fig.suptitle('M=1.2 M$_\odot$: best fit', fontsize=16)
    elif iteration == 3:
        fig.suptitle('M=1.2 M$_\odot$: MESA comparison', fontsize=16)

    plt.tight_layout()
    if iteration == 1:
        plt.savefig('figures/separate_initial.pdf', bbox_inches='tight')
    elif iteration == 2:
        plt.savefig('figures/separate_BestFit.pdf', bbox_inches='tight')
    elif iteration == 3:
        plt.savefig('figures/MESA_comparison.pdf', bbox_inches='tight')
    plt.show()


def plot_together(sol_cen, sol_surf, M, iteration):
    '''
        plot normalized stellar parameters together as a function of solar mass
        inputs:
            sol_cen: array
                solution from outward integrations (center out)
            sol_surf: array
                solution from inward integrations (surface in)
            M: float
                mass of star, used for normalization
            iteration: int
                1 or 2 (intial guess vs best-fit)
        returns:
            none, generates/saves plots
    '''
    color = ['C2', 'C4', 'C6', 'C9']
    label = ['Luminosity [erg/s]', 'Pressure [dyne/cm$^{2}$]', 'Radius [cm]', 'Temperature [K]']
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    for i in range(4):
        scale = np.abs(sol_surf.y[i, 0] - sol_cen.y[i, 0])  # scale factor is difference between the two solutions
        zero_point = np.min([sol_surf.y[i, 0], sol_cen.y[i, 0]])  # use min solution as zero point for normalization

        ax.plot(sol_cen.t/M, (sol_cen.y[i] - zero_point) / scale, label=label[i], color=color[i])
        ax.plot(sol_surf.t/M, (sol_surf.y[i] - zero_point) / scale, color=color[i])

    ax.legend(fontsize="6")
    ax.set_xlabel('M/M$_\odot$')
    ax.set_ylabel('Normalized Quantity')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if iteration == 1:
        ax.set_title('M=1.2 M$_\odot$: initial guess')
    elif iteration == 2:
        ax.set_title('M=1.2 M$_\odot$: best fit')
    fig.tight_layout()
    if iteration == 1:
        plt.savefig('figures/normalized_initial.pdf', bbox_inches='tight')
    elif iteration == 2:
        plt.savefig('figures/normalized_BestFit.pdf', bbox_inches='tight')

    plt.show()


def mesa_vals():
    '''
    calculates the MESA luminosity from the values in the csv file
    inputs:
        none
    returns:
        L_mesa * c.Ms: array
            MESA luminosity values [erg/s]
        p: mesa_reader.MesaData
            MESA values
    '''
    #read in profiles using mesa_reader
    data = mr.MesaLogDir(log_path='./LOGS')
    p = data.profile_data()

    eps = p.pp+p.cno  # energy generation rate (combo of pp-chain/cno)
    L_mesa = np.zeros(len(p.pp))

    for i in range(len(p.pp) - 1, -1, -1):
        if i == len(p.pp) - 1: # last data point
            L_mesa[i] = eps[i] * p.mass[i]
        else: # calculate luminosity for current step, add to previous step
            L_mesa[i] = eps[i] * (p.mass[i] - p.mass[i + 1]) + L_mesa[i + 1]

    return L_mesa * c.Ms, p


def save_csv(sol_cen, sol_surf):
    '''
    saves quantities of interest to a csv file
    inputs:
        sol_cen: array
            solution from outward integrations (center out)
        sol_surf: array
            solution from inward integrations (surface in)
    returns:
        none, generates/saves csv file
    '''
    # reverse surface values to match center values
    reversed_surf = sol_surf.y[:, ::-1]
    reversed_surf_m = sol_surf.t[::-1]

    # combine center and surface results
    L = np.concatenate((sol_cen.y[0, :], reversed_surf[0, :]))
    P = np.concatenate((sol_cen.y[1, :], reversed_surf[1, :]))
    R = np.concatenate((sol_cen.y[2, :], reversed_surf[2, :]))
    T = np.concatenate((sol_cen.y[3, :], reversed_surf[3, :]))
    M = np.concatenate((sol_cen.t, reversed_surf_m))

    rho = density.density(P, T)
    epsilon = energy.pp_rate(rho, T) + energy.cno_rate(rho, T)
    kappa = opacity.opacity(T, rho)
    del_rad = energy.del_val(M, L, P, kappa, T, ad_v_rad='rad')
    del_ad = np.ones(np.shape(del_rad)) * 0.4

    nabla_list = []
    rad_v_conv = []
    for i in range(len(L)):
        nabla = energy.del_val(M[i], L[i], P[i], kappa[i], T[i])
        nabla_list.append(nabla)

        if nabla >= 0.4:
            rad_v_conv.append('convective')
        else:
            rad_v_conv.append('radiative')

    # create dataframe to save as csv
    params = np.asarray([M, R, rho, T, P, L, epsilon, kappa, del_ad, del_rad, nabla_list, rad_v_conv])
    headers = ['mass', 'radius', 'density', 'temp', 'pressure', 'luminosity', 'epsilon', 'kappa', 'del_ad', 'del_rad',
               'nabla', 'rad. vs. conv.']
    df = pd.DataFrame(np.transpose(params), columns=headers)
    df.to_csv('stellar_parameters.csv')


def percent_dif(sol_cen, sol_surf):
    '''
    calculates percent difference between model values and MESA values
    inputs:
        sol_surf: array
            solution from outward integrations (center out)
    returns:
        nothing, prints percent difference
    '''
    L_mesa, p = mesa_vals()

    # calculate surface gravity, g = GM/R^2
    g_mesa = c.G*(p.mass[0]*c.Ms) / (p.R[0]*c.Rs)**2
    g_model = c.G * (sol_surf.t[0]) / (sol_surf.y[2, 0])**2

    # percent difference = abs(difference/average)*100 = abs((x2-x1) / ((x2+x1)/2))*100
    L_dif = np.abs((L_mesa[0] - sol_surf.y[0, 0]) / ((L_mesa[0] + sol_surf.y[0, 0]) / 2)) * 100
    P_dif = np.abs((p.P[-1] - sol_cen.y[1, 0]) / ((p.P[-1] + sol_cen.y[1, 0]) / 2)) * 100
    R_dif = np.abs((p.R[0] * c.Rs - sol_surf.y[2, 0]) / ((p.R[0] * c.Rs + sol_surf.y[2, 0]) / 2)) * 100
    T_dif = np.abs((p.T[0] - sol_surf.y[3, 0]) / ((p.T[0] + sol_surf.y[3, 0]) / 2)) * 100
    g_dif = np.abs((g_mesa - g_model) / ((g_mesa + g_model) / 2)) * 100

    print('Percent Difference:')
    print('Luminosity: {:.2f}'.format(L_dif))
    print('Pressure: {:.2f}'.format(P_dif))
    print('Radius: {:.2f}'.format(R_dif))
    print('Temperature: {:.2f}'.format(T_dif))
    print('Surface Gravity: {:.2f}'.format(g_dif))


