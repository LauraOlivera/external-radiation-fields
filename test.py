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


# Define the frecuency cutoff for the x-ray corona spectral shape
nu_c = ((150*u.keV)/const.h).to('Hz')

# Define an energy grid and a consequent frequency grid
energy_grid_ext = np.logspace(-13, 0,131)*u.GeV
freq_grid_ext = energy_grid_ext.to(u.Hz, equivalencies=u.spectral())

#energy_grid = np.logspace(-13, -3,101)*u.GeV
#freq_grid = energy_grid.to(u.Hz, equivalencies=u.spectral())

# The parameters of test follow the discussion in Ghisellini and Tavecchio 2009
#test = exrad.ext_rad(1e9*const.M_sun,L_disk = 1e45*u.erg*u.s**-1,f_x = 0.1,f_BLR = 0.1,f_BLR_x = 0.01,f_IR = 0.5,R_in_s = 3,R_out_s = 500,R_x_s = 30,eta = 0.08,alpha_x = 1,nu_c = nu_c,gamma = 15,z=0)

# Parameters of TXS 0506+956 as described in Padovani et al 2019
TXS = exrad.ext_rad(3e8*const.M_sun,L_disk = 8.5e44*u.erg*u.s**-1,f_x = 0.1,f_BLR = (5e43)/(8.5e44),f_BLR_x = 0.1*(5e43)/(8.5e44),f_IR = 0.5,R_in_s = 3,R_out_s = 500,R_x_s = 30,eta = 0.08,alpha_x = 1,nu_c = nu_c,gamma = 15,z=0)

# Define some distances from the black hole
R_diss = [1e1,1e2,1e3,1e4, 1e5,1e6]*TXS.R_s

# Compute the transformed fields for that distance
transformed_fields1 = TXS.transform_external_fields(freq_grid_ext,1e2*TXS.R_s)
##transformed_fields2 = TXS.transform_external_fields(freq_grid_ext, R_diss[1])
##transformed_fields3 = TXS.transform_external_fields(freq_grid_ext, R_diss[2])
##transformed_fields4 = TXS.transform_external_fields(freq_grid_ext, R_diss[3])
##transformed_fields5 = TXS.transform_external_fields(freq_grid_ext, R_diss[4])
##transformed_fields6 = TXS.transform_external_fields(freq_grid_ext, R_diss[5])

# Plot
fig1 = plt.figure(figsize=(8,6))

plt.loglog(freq_grid_ext, freq_grid_ext*transformed_fields1[0], label = 'corona+disk', color='firebrick')
plt.loglog(freq_grid_ext, freq_grid_ext*transformed_fields1[1], label = 'BLR', linestyle = '--', color='navy')
plt.loglog(freq_grid_ext, freq_grid_ext*transformed_fields1[2], label = 'IR torus', linestyle = ':', color='k')

##plt.loglog(freq_grid_ext, freq_grid_ext*transformed_fields2[0], color='firebrick')
##plt.loglog(freq_grid_ext, freq_grid_ext*transformed_fields2[1], linestyle = '--', color='navy')
##plt.loglog(freq_grid_ext, freq_grid_ext*transformed_fields2[2], linestyle = ':', color='k')

##plt.loglog(freq_grid_ext, freq_grid_ext*transformed_fields3[0], color='firebrick')
##plt.loglog(freq_grid_ext, freq_grid_ext*transformed_fields3[1], linestyle = '--', color='navy')
##plt.loglog(freq_grid_ext, freq_grid_ext*transformed_fields3[2], linestyle = ':', color='k')

##plt.loglog(freq_grid_ext, freq_grid_ext*transformed_fields4[0], color='firebrick')
##plt.loglog(freq_grid_ext, freq_grid_ext*transformed_fields4[1], linestyle = '--', color='navy')
##plt.loglog(freq_grid_ext, freq_grid_ext*transformed_fields4[2], linestyle = ':', color='k')

##plt.loglog(freq_grid_ext, freq_grid_ext*transformed_fields5[0], color='firebrick')
##plt.loglog(freq_grid_ext, freq_grid_ext*transformed_fields5[1], linestyle = '--', color='navy')
##plt.loglog(freq_grid_ext, freq_grid_ext*transformed_fields5[2], linestyle = ':', color='k')

##plt.loglog(freq_grid_ext, freq_grid_ext*transformed_fields6[0], color='firebrick')
##plt.loglog(freq_grid_ext, freq_grid_ext*transformed_fields6[1], linestyle = '--', color='navy')
##plt.loglog(freq_grid_ext, freq_grid_ext*transformed_fields6[2], linestyle = ':', color='k')


plt.legend()
plt.title('$\Gamma$ ='+ str(TXS.gamma) + ', $R_{diss}$ =  10$^1$R$_s$')
plt.ylabel(r'Log $\nu^{\prime}$U$^{\prime}_{\nu^{\prime}}$ [erg cm$^-3$]')
plt.xlabel(r'Log $\nu^{\prime}$ [Hz]')
plt.ylim(1e-14, 1e6)
plt.xlim(1e11, 1e23)
plt.yticks(np.logspace(-14,6,21))
plt.xticks(np.logspace(11,23,13))
plt.grid()

plt.show()



