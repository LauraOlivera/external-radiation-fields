import external_radiation as exrad
import numpy as np
import matplotlib.pyplot as plt
import astropy
from matplotlib.pyplot import cm
from astropy import constants as const
from astropy import units as u
import numpy.ma as ma
import scipy.integrate as integrate
import csv
from scipy.integrate import simps

nu_c = ((150*u.keV)/const.h).to('Hz')


energy_grid_ext = np.logspace(-13, 0,131)*u.GeV
freq_grid_ext = energy_grid_ext.to(u.Hz, equivalencies=u.spectral())

energy_grid = np.logspace(-13, -3,101)*u.GeV
freq_grid = energy_grid.to(u.Hz, equivalencies=u.spectral())


test = exrad.ext_rad(1e9*const.M_sun,L_disk = 1e45*u.erg*u.s**-1,f_x = 0.1,f_BLR = 0.1,f_BLR_x = 0.01,f_IR = 0.5,R_in_s = 3,R_out_s = 500,R_x_s = 30,eta = 0.08,alpha_x = 1,nu_c = nu_c,gamma = 15,z=0)
R_diss = 1e3*test.R_s

#print(test.final_fun_BLR_bb(0,freq_grid,R_diss))

#plt.loglog(freq_grid, freq_grid*test.u_d_logR(freq_grid, R_diss)[0], label = 'disk1')
#plt.loglog(freq_grid, freq_grid*test.u_d_R(freq_grid, R_diss)[0], label = 'disk1')
#plt.loglog(freq_grid, freq_grid*test.u_d_mu(freq_grid, R_diss)[0], label = 'disk1')

#plt.loglog(freq_grid_ext, freq_grid_ext*test.u_x_logR(freq_grid_ext, R_diss)[0], label = 'corona3')
#plt.loglog(freq_grid_ext, freq_grid_ext*test.u_x_R(freq_grid_ext, R_diss)[0], label = 'corona2')

plt.loglog(freq_grid_ext, freq_grid_ext*test.u_IR(freq_grid_ext, R_diss)[0], label = 'IR torus', linestyle = ':')
plt.loglog(freq_grid_ext, freq_grid_ext*exrad.ext_rad.join_disk_corona(test.u_d_logR(freq_grid, R_diss)[0],test.u_x_mu(freq_grid_ext, R_diss)[0],freq_grid_ext), label = 'corona+disk')
plt.loglog(freq_grid_ext, freq_grid_ext*exrad.ext_rad.join_disk_corona(test.u_BLR_bb(freq_grid, R_diss)[0],test.u_BLR_x(freq_grid_ext, R_diss)[0],freq_grid_ext), label = 'BLR', linestyle = '--')

#plt.loglog(freq_grid_ext, freq_grid_ext*test.u_BLR_x(freq_grid_ext, R_diss)[0], label = 'BLR1')


plt.legend()
plt.title('Spectra of radiation energy density in the comoving frame')
plt.ylabel(r'Log $\nu^{\prime}$U$^{\prime}_{\nu^{\prime}}$ [erg cm$^-3$]')
plt.xlabel(r'Log $\nu^{\prime}$ [Hz]')
plt.ylim(1e-14, 1e5)
plt.xlim(1e11, 1e20)
plt.yticks(np.logspace(-14,5,20))
plt.xticks(np.logspace(11,23,13))
plt.grid()

plt.show()



