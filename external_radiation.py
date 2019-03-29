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


class ext_rad(object):
	""" Class for computing the comoving frame energy densities of photon fields
	in the surroundings of a blazar"""

	def __init__(self,M_BH, L_disk,f_x,f_BLR, f_BLR_x,f_IR,R_in_s,R_out_s,R_x_s, eta,alpha_x, nu_c,gamma,z):
		""" The instance of the class will have the relevant parameters of the source necessary to compute these external photon fields.

		############################################################################################

		Input parameters:
		-----------------
		M_BH : `~astropy.units.Quantity`
			mass of the central black hole
		L_disk : `~astropy.units.Quantity`
			luminosity of the accretion disk
		f_x : float
			fraction of the disk luminosity emitted by the x-ray corona
		f_BLR : float
			percentaje of disk luminosity reprocessed by Broad Line Region
		f_BLR_x : float
			percentaje of x-ray corona luminosity reprocessed by Broad Line Region
		f_IR : float
			percentaje of disk luminosity reprocessed by the IR torus
		R_in_s : float
			lower limit for the disk extension, in Schwarzschild radius
		R_out_s : float
			upper limit for the disk extension, in Schwarzschild radius
		R_x_s : float
			upper limit for the corona extension, in Schwarzschild radius
		alpha_x : float
			index of the x-ray power law
		nu_c : '~astropy.units.Quantity'
			Cutoff frequency of the x-ray power law
		eta : float
			efficiency of the accretion
		gamma : float
			Lorentz factor of the blob motion
		z : float
			redshift of the source

		###########################################################################################
		
		Secondary quantities:
		---------------------
		R_s : `~astropy.units.Quantity`
			Schwarzschild radius
		R_in : `~astropy.units.Quantity`
			lower limit for the disk extension, in cm
		R_out : `~astropy.units.Quantity`
			upper limit for the disk extension, in cm
		R_x : `~astropy.units.Quantity`
			upper limit for the corona extension, in cm
		R_BLR : `~astropy.units.Quantity`
			radius of the Broad Line Region, in cm, as in P1
		R_IR : `~astropy.units.Quantity`
			radius of the IR torus, in cm, as in P1
		T_BLR_rest : `~astropy.units.Quantity`
			black hole rest frame temperature of the blackbody spectra with which we approximate the BLR as in P1
		T_IR_rest : `~astropy.units.Quantity`
			black hole rest frame temperature of the blackbody spectra with which we approximate the IR torus as in P1

		###########################################################################################
		
		Main functions:
		---------------
		transform_external_fields(self,freq, R_diss): given a frequency grid and a distance from the central black hole,
								it returns the blob frame energy density of all three components.
		
		###########################################################################################
		
		References:
		-----------
		P1 : Canonical high-power blazars, G. Ghisellini and F. Tavecchio, MNRAS 397, (2009)
		P2 : The origin of gamma-ray emission in blazars, G. Ghisellini and P. Madau, MNRAS 280, (1996)
		"""

		self.M_BH = M_BH
		self.L_disk = L_disk
		self.f_x = f_x
		self.f_BLR = f_BLR
		self.f_BLR_x = f_BLR_x
		self.f_IR = f_IR
		self.R_in_s = R_in_s
		self.R_out_s = R_out_s
		self.eta = eta
		self.alpha_x = alpha_x
		self.nu_c = nu_c
		self.gamma = gamma
		self.z = z

		# Define the Schwarzschild radius and the disk and corona extension
		self.R_s = ((2*const.G*M_BH)/(const.c**2)).to(u.cm)
		self.R_in = R_in_s*self.R_s
		self.R_out = R_out_s*self.R_s
		self.R_x = R_x_s*self.R_s
		
		# Define the radius of the BLR and IR torus
		self.R_BLR = 1e17*np.sqrt((self.L_disk.to('erg s-1')).value*1e-45)*u.cm
		self.R_IR = 2.5e18*np.sqrt((self.L_disk.to('erg s-1')).value*1e-45)*u.cm

		# Define the rest temperature of the BLR and IR torus 
		self.T_BLR_rest = (const.h*(2.47e15*u.Hz)/(3.93*const.k_B)).to('K')
		self.T_IR_rest = 370*u.K



	def T_disk(self,R):
		"""Given the radius within the disk, it returns the temperature of the blackbody emitted by that annulus.
		The formula used is (1) from P1.
		input : `~astropy.units.Quantity`
		returns : `~astropy.units.Quantity`"""
		a = (3*self.R_s*self.L_disk)/(16*np.pi*self.eta*const.sigma_sb*(R**3))
		b = (1-np.sqrt(3*self.R_s/R))
		T = (a*b)**(1/4)
		return T.to('K')

	def mu_d(R, R_diss):
		"""Given a radius within the disk or corona, it returns the cosine of the angle at which the blob sees said radius
		input : `~astropy.units.Quantity`
		returns : `~astropy.units.Quantity`"""
		mu = np.sqrt(1+((R.to(u.cm))**2/(R_diss.to(u.cm))**2))**-1
		return mu

	def bb_nu(nu,T):
		"""Given a temperature and an array of frequencies, it returns the black-body intensity of a blackbody of that temperature
		input : ('~astropy.units.Quantity',~astropy.units.Quantity')
		returns : '~astropy.units.Quantity'"""
		a = ((2*const.h*(nu**3))/(const.c**2))
		b = (np.exp((const.h*nu)/(const.k_B*T))-1)**-1
		B = (a*b*u.sr**-1).to('W sr-1 m-2 Hz-1')
		return B

	def xray_lum(nu,Lo_x,nu_c, alpha_x):
		""" x-ray corona spectra, give, by a power law with a cuttoff in nu_c) """
		return Lo_x*nu**(-alpha_x)*np.exp(-nu/nu_c)

	def norm_x(self,nu):
		""" Given a frequency range and for the defined corona luminosity, it returns the normalization factor 
		of the x-ray corona spectral luminosity """
		L_x = self.f_x*self.L_disk
		integral_x = simps(ext_rad.xray_lum(nu,1*u.erg*u.s**-1,self.nu_c, self.alpha_x).value, nu.value)
		Lo_x = (L_x.value/integral_x)*u.erg*u.s**-1
		return Lo_x

	def beta(gamma):
		""" From the bulk Lorentz factor, compute the ratio of the velocity to c"""
		a = (1 - 1/gamma)*(1 + 1/gamma)
		beta = np.sqrt(a)
		return beta

	def doppler(gamma,mu):
		""" Relativistic Doppler factor"""
		d = (gamma*(1-ext_rad.beta(gamma)*mu))**-1
		return d

	###################################################################################################################################
	### Functions related to the accretion disk transformation. We can perform the integration in mu (cos of the angle), R and logR ###
	###################################################################################################################################
	
	# Integration in mu
	def final_fun(self,mu,freq,R_diss):
		""" Compute the integrand in mu of the transformation to comoving frame of the energy density of the blackbody
		as in (17) from P1. Note that a blackbody only transforms via the temperature """
		R=(R_diss.to(u.cm)/mu)*np.sqrt(1-mu**2)
		d = ext_rad.doppler(self.gamma,mu)
		bb = ext_rad.bb_nu(freq, ext_rad.T_disk(self,R)*(d**-1))
		prod = (d**2)*bb
		return prod.to('erg s-1 sr-1 cm-2 Hz-1')

	def u_d_mu(self,freq, R_diss):
		""" Returns the transformed energy density (result, error) of the disk at a distance from the blackbody R_diss """
		result = [integrate.quad(lambda x: ext_rad.final_fun(self,x,i,R_diss).value,ext_rad.mu_d(self.R_out.to(u.cm),R_diss.to(u.cm)), ext_rad.mu_d(self.R_in.to(u.cm),R_diss.to(u.cm)))[0] for i in freq]*u.erg*u.s**-1*u.cm**-2*u.Hz**-1
		error = [integrate.quad(lambda x: ext_rad.final_fun(self,x,i,R_diss).value,ext_rad.mu_d(self.R_out.to(u.cm),R_diss.to(u.cm)), ext_rad.mu_d(self.R_in.to(u.cm),R_diss.to(u.cm)))[1] for i in freq]*u.erg*u.s**-1*u.cm**-2*u.Hz**-1
		result_fin = (((2*np.pi)/(const.c))*result).to('erg cm-3 Hz-1')
		error_fin = (((2*np.pi)/(const.c))*error).to('erg cm-3 Hz-1')
		return result_fin, error_fin


	# Integration in R
	def final_fun_R(self, R,freq,R_diss):
		""" Compute the integrand in R of the transformation to comoving frame of the energy density of the blackbody """
		mu = np.sqrt(1+((R.to(u.cm))**2/(R_diss.to(u.cm))**2))**-1
		d = ext_rad.doppler(self.gamma,mu)
		bb = ext_rad.bb_nu(freq, ext_rad.T_disk(self,R)*(d**-1))
		prod = (d**2)*bb
		return prod.to('erg s-1 sr-1 cm-2 Hz-1')

	def u_d_R(self,freq, R_diss):
		""" Returns the transformed energy density (result, error) of the disk at a distance from the blackbody R_diss """
		result = [integrate.quad(lambda x: ((((ext_rad.mu_d(x*u.cm,R_diss.to(u.cm)))**2)*np.sqrt(1-ext_rad.mu_d(x*u.cm,R_diss.to(u.cm))))*(R_diss**-1)*ext_rad.final_fun_R(self,x*u.cm,i,R_diss)).value,(self.R_in.to(u.cm)).value, (self.R_out.to(u.cm)).value)[0] for i in freq]*u.erg*u.s**-1*u.cm**-2*u.Hz**-1
		error = [integrate.quad(lambda x: ((((ext_rad.mu_d(x*u.cm,R_diss.to(u.cm)))**2)*np.sqrt(1-ext_rad.mu_d(x*u.cm,R_diss.to(u.cm))))*(R_diss**-1)*ext_rad.final_fun_R(self,x*u.cm,i,R_diss)).value,(self.R_in.to(u.cm)).value, (self.R_out.to(u.cm)).value)[1] for i in freq]*u.erg*u.s**-1*u.cm**-2*u.Hz**-1
		result_fin = (((2*np.pi)/(const.c))*result).to('erg cm-3 Hz-1')
		error_fin = (((2*np.pi)/(const.c))*error).to('erg cm-3 Hz-1')
		return result_fin, error_fin

	# Integration in logR
	def final_fun_logR(self,logR,freq,R_diss):
		R = 10**(logR)*u.cm
		mu = np.sqrt(1+((R.to(u.cm))**2/(R_diss.to(u.cm))**2))**-1
		d = ext_rad.doppler(self.gamma,mu)
		bb = ext_rad.bb_nu(freq, ext_rad.T_disk(self,R)*(d**-1))
		prod = (d**2)*bb
		return prod.to('erg s-1 sr-1 cm-2 Hz-1')

	def u_d_logR(self,freq, R_diss):
		x_v = np.linspace(np.log10(self.R_in.to(u.cm).value), np.log10(self.R_out.to(u.cm).value), 101)
		result = [integrate.quad(lambda x: ((((ext_rad.mu_d((10**x)*u.cm,R_diss.to(u.cm)))**2)*np.sqrt(1-ext_rad.mu_d((10**x)*u.cm,R_diss.to(u.cm))))*(R_diss)**-1*np.log(10)*(10**x)*ext_rad.final_fun_logR(self,x,i,R_diss)).value,x_v[0], x_v[-1])[0] for i in freq]*u.erg*u.s**-1*u.cm**-2*u.Hz**-1
		error = [integrate.quad(lambda x: ((((ext_rad.mu_d((10**x)*u.cm,R_diss.to(u.cm)))**2)*np.sqrt(1-ext_rad.mu_d((10**x)*u.cm,R_diss.to(u.cm))))*(R_diss)**-1*np.log(10)*(10**x)*ext_rad.final_fun_logR(self,x,i,R_diss)).value,x_v[0], x_v[-1])[1] for i in freq]*u.erg*u.s**-1*u.cm**-2*u.Hz**-1
		result_fin = (((2*np.pi)/(const.c))*result).to('erg cm-3 Hz-1')
		error_fin = (((2*np.pi)/(const.c))*error).to('erg cm-3 Hz-1')
		return result_fin, error_fin


	###################################################################################################################################
	### Functions related to the corona transformation. We can perform the integration in mu (cos of the angle), R and logR ###
	###################################################################################################################################		

	# Integration in mu
	def final_function_x_mu(self,mu,freq,R_diss,Lo_x):
		R_x=(R_diss.to(u.cm)/mu)*np.sqrt(1-mu**2)
		d = ext_rad.doppler(self.gamma,mu)
		tr_I_x = ((2*np.pi*np.pi*(R_x**2)*u.sr)**-1)*ext_rad.xray_lum(freq,Lo_x*d**-1,self.nu_c*d**-1, self.alpha_x)
		prod = tr_I_x*d**-1
		return prod.to('erg s-1 sr-1 cm-2 Hz-1')

	def u_x_mu(self,freq,R_diss):
		Lo_x = ext_rad.norm_x(self, freq)
		result_x = [integrate.quad(lambda x: ext_rad.final_function_x_mu(self,x,i,R_diss,Lo_x).value,ext_rad.mu_d(self.R_x.to(u.cm),R_diss.to(u.cm)), ext_rad.mu_d(self.R_in.to(u.cm),R_diss.to(u.cm)))[0] for i in freq]*u.erg*u.s**-1*u.cm**-2*u.Hz**-1
		error_x = [integrate.quad(lambda x: ext_rad.final_function_x_mu(self,x,i,R_diss,Lo_x).value,ext_rad.mu_d(self.R_x.to(u.cm),R_diss.to(u.cm)), ext_rad.mu_d(self.R_in.to(u.cm),R_diss.to(u.cm)))[1] for i in freq]*u.erg*u.s**-1*u.cm**-2*u.Hz**-1
		error_x_fin = (((2*np.pi)/(const.c))*error_x).to('erg cm-3 Hz-1')
		result_x_fin = (((2*np.pi)/(const.c))*result_x).to('erg cm-3 Hz-1')
		return result_x_fin, error_x_fin

	# Integration in R
	def final_fun_x_R(self,R_x,freq,R_diss,Lo_x):
		mu = np.sqrt(1+((R_x.to(u.cm))**2/(R_diss.to(u.cm))**2))**-1
		d = ext_rad.doppler(self.gamma,mu)
		tr_I_x = ((2*np.pi*np.pi*(R_x**2)*u.sr)**-1)*ext_rad.xray_lum(freq,Lo_x*d**-1,self.nu_c*d**-1, self.alpha_x)
		prod = tr_I_x*d**-1
		return prod.to('erg s-1 sr-1 cm-2 Hz-1')

	def u_x_R(self,freq, R_diss):
		Lo_x = ext_rad.norm_x(self, freq)
		result_x = [integrate.quad(lambda x: ((((ext_rad.mu_d(x*u.cm,R_diss.to(u.cm)))**2)*np.sqrt(1-ext_rad.mu_d(x*u.cm,R_diss.to(u.cm))))*(R_diss**-1)*ext_rad.final_fun_x_R(self,x*u.cm,i,R_diss,Lo_x)).value,(self.R_in.to(u.cm)).value, (self.R_x.to(u.cm)).value)[0] for i in freq]*u.erg*u.s**-1*u.cm**-2*u.Hz**-1
		error_x = [integrate.quad(lambda x: ((((ext_rad.mu_d(x*u.cm,R_diss.to(u.cm)))**2)*np.sqrt(1-ext_rad.mu_d(x*u.cm,R_diss.to(u.cm))))*(R_diss**-1)*ext_rad.final_fun_x_R(self,x*u.cm,i,R_diss,Lo_x)).value,(self.R_in.to(u.cm)).value, (self.R_x.to(u.cm)).value)[1] for i in freq]*u.erg*u.s**-1*u.cm**-2*u.Hz**-1
		result_x_fin = (((2*np.pi)/(const.c))*result_x).to('erg cm-3 Hz-1')
		error_x_fin = (((2*np.pi)/(const.c))*error_x).to('erg cm-3 Hz-1')
		return result_x_fin, error_x_fin

	# Integration in logR
	def final_fun_x_logR(self,logR,freq,R_diss,Lo_x):
		R_x = 10**(logR)*u.cm
		mu = np.sqrt(1+((R_x.to(u.cm))**2/(R_diss.to(u.cm))**2))**-1
		d = ext_rad.doppler(self.gamma,mu)
		tr_I_x = ((2*np.pi*np.pi*(R_x**2)*u.sr)**-1)*ext_rad.xray_lum(freq,Lo_x*d**-1,self.nu_c*d**-1, self.alpha_x)
		prod = tr_I_x*d**-1
		return prod.to('erg s-1 sr-1 cm-2 Hz-1')

	def u_x_logR(self, freq, R_diss):
		Lo_x = ext_rad.norm_x(self, freq)
		x_v = np.linspace(np.log10(self.R_in.to(u.cm).value), np.log10(self.R_x.to(u.cm).value), 101)
		result_x = [integrate.quad(lambda x: ((((ext_rad.mu_d((10**x)*u.cm,R_diss.to(u.cm)))**2)*np.sqrt(1-ext_rad.mu_d((10**x)*u.cm,R_diss.to(u.cm))))*(R_diss)**-1*np.log(10)*(10**x)*ext_rad.final_fun_x_logR(self,x,i,R_diss,Lo_x)).value,x_v[0], x_v[-1])[0] for i in freq]*u.erg*u.s**-1*u.cm**-2*u.Hz**-1
		error_x = [integrate.quad(lambda x: ((((ext_rad.mu_d((10**x)*u.cm,R_diss.to(u.cm)))**2)*np.sqrt(1-ext_rad.mu_d((10**x)*u.cm,R_diss.to(u.cm))))*(R_diss)**-1*np.log(10)*(10**x)*ext_rad.final_fun_x_logR(self,x,i,R_diss,Lo_x)).value,x_v[0], x_v[-1])[1] for i in freq]*u.erg*u.s**-1*u.cm**-2*u.Hz**-1
		result_x_fin = (((2*np.pi)/(const.c))*result_x).to('erg cm-3 Hz-1')
		error_x_fin = (((2*np.pi)/(const.c))*error_x).to('erg cm-3 Hz-1')
		return result_x_fin, error_x_fin

	# Join both corona and disk to plot
	def join_disk_corona(u_d, u_x, freq):
		""" Input the transformed energy densities of both components, output the merged curve """
		u_x[:np.argmax(u_d)+5]=np.nan
		u_d_extended = np.zeros(len(freq))
		u_d_extended[:len(u_d)]=u_d
		A = u_x
		B = u_d_extended
		both = np.nansum(np.stack((A,B)), axis=0)*u.erg*u.cm**-3*u.Hz**-1
		return both


	####################################################################################################################################
	### Functions related to the BLR transformation. We will perform the integration in cos(alpha), the incident angle from the disk ###
	####################################################################################################################################


	# Define the normalization of the blackbody: we will normalize using the energy density for R_diss<R_BLR and the luminosity for R_BLR>R_diss

	def normalization_BLR(self,R_diss):
		""" normalization of the BLR blackbody: we will normalize using the energy density for R_diss<R_BLR and the luminosity for R_BLR>R_diss"""
		if (R_diss.to('cm')).value < (3*self.R_BLR).value:
			u1 = ((self.f_BLR*self.L_disk)/(4*np.pi*self.R_BLR**2*const.c)).to('erg cm-3')
			u2 = ((4*const.sigma_sb*self.T_BLR_rest**4)/(const.c)).to('erg cm-3')
			norm = u1/u2    
		else:
			norm = (self.f_BLR*self.L_disk)/((4*np.pi*(self.R_BLR**2)*const.sigma_sb*self.T_BLR_rest**4).to('erg s-1'))
		return norm


	# Implement the relation between the cosine of theta (angle at which the blob sees the BLR) and cosine of alpha (angle of which the BLR sees the disk)
	def angle_inv(R_BLR,cos_alpha,R_diss):
		""" relation between the cosine of theta (angle at which the blob sees the BLR) and cosine of alpha (angle of which the BLR sees the disk). Note that it is different for the case where the blob is inside and outside of the BLR."""
		x = 2*R_BLR*R_diss*cos_alpha-R_diss**2-(R_BLR**2)*(cos_alpha**2)
		y = 2*R_BLR*R_diss*cos_alpha-R_diss**2-(R_BLR**2)
		if (R_diss.to('cm')).value < (R_BLR.to('cm')).value:
			a = -np.sqrt(x/y)
		else:
			a = np.sqrt(x/y)
		return a
	
	# Reprocessed disk radiation: integrate the blackbody component in cos(alpha) that goes from 0 to 1

	def final_fun_BLR_bb(self,cos_alpha,freq,R_diss):
		mu = ext_rad.angle_inv(self.R_BLR,cos_alpha,R_diss)
		d = ext_rad.doppler(self.gamma,mu)
		bb = ext_rad.normalization_BLR(self,R_diss)*ext_rad.bb_nu(freq, self.T_BLR_rest*d**-1)*cos_alpha
		diff = (self.R_BLR**2*(2*cos_alpha*(R_diss**2+self.R_BLR**2)-2*self.R_BLR*R_diss*(cos_alpha**2+1)))/(2*(2*self.R_BLR*R_diss*cos_alpha-self.R_BLR**2-R_diss**2)**2*np.sqrt((-2*self.R_BLR*R_diss*cos_alpha+(R_diss**2)+(self.R_BLR**2)*(cos_alpha**2))/(-2*self.R_BLR*R_diss*cos_alpha+R_diss**2+(self.R_BLR**2))))
		prod = (d**2)*bb
		return (prod.to('erg s-1 sr-1 cm-2 Hz-1'))*np.abs(diff)


	def u_BLR_bb(self,freq, R_diss):
		result = [integrate.quad(lambda x: ext_rad.final_fun_BLR_bb(self,x,i,R_diss).value,0, 1)[0] for i in freq]*u.erg*u.s**-1*u.cm**-2*u.Hz**-1
		error = [integrate.quad(lambda x: ext_rad.final_fun_BLR_bb(self,x,i,R_diss).value,0, 1)[1] for i in freq]*u.erg*u.s**-1*u.cm**-2*u.Hz**-1
		result_fin = (((2*np.pi)/(const.c))*result).to('erg cm-3 Hz-1')
		error_fin = (((2*np.pi)/(const.c))*error).to('erg cm-3 Hz-1')
		return result_fin, error_fin



	# Reprocessed corona radiation: same thing but with the x-ray luminosity. SO FAR WE ARE TREATING DIFFERENTLY THE CASES FOR R_diss>R_BLR and viceversa. MAKE SURE THAT THIS IS OK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	def final_function_BLR_x(self,cos_alpha,freq,R_diss,Lo_x):
		mu = ext_rad.angle_inv(self.R_BLR,cos_alpha,R_diss)
		d = ext_rad.doppler(self.gamma,mu)
		# In the paper they use 2pi, I think 4pi is more correct: VERIFY
		tr_I_BLR_x = (((4*np.pi*(self.R_BLR**2)*u.sr)**-1)*self.f_BLR_x*ext_rad.xray_lum(freq,Lo_x*d**-1,self.nu_c*d**-1, self.alpha_x)*cos_alpha).to('erg cm-2 Hz-1 s-1 sr-1')   
		diff = (self.R_BLR**2*(2*cos_alpha*(R_diss**2+self.R_BLR**2)-2*self.R_BLR*R_diss*(cos_alpha**2+1)))/(2*(2*self.R_BLR*R_diss*cos_alpha-self.R_BLR**2-R_diss**2)**2*np.sqrt((-2*self.R_BLR*R_diss*cos_alpha+(R_diss**2)+(self.R_BLR**2)*(cos_alpha**2))/(-2*self.R_BLR*R_diss*cos_alpha+R_diss**2+(self.R_BLR**2))))
		if (R_diss.to('cm')).value < (3*self.R_BLR.to('cm')).value:
			prod = tr_I_BLR_x*d**-1*(4*np.pi)**-1
		else:
			prod = tr_I_BLR_x*d**-1            
		return (prod.to('erg s-1 sr-1 cm-2 Hz-1'))*np.abs(diff)

	def u_BLR_x(self,freq,R_diss):
		Lo_x = ext_rad.norm_x(self, freq)
		result_x = [integrate.quad(lambda x: ext_rad.final_function_BLR_x(self,x,i,R_diss,Lo_x).value,0, 1)[0] for i in freq]*u.erg*u.s**-1*u.cm**-2*u.Hz**-1
		error_x = [integrate.quad(lambda x: ext_rad.final_function_BLR_x(self,x,i,R_diss,Lo_x).value,0, 1)[1] for i in freq]*u.erg*u.s**-1*u.cm**-2*u.Hz**-1
		error_x_fin = (((2*np.pi)/(const.c))*error_x).to('erg cm-3 Hz-1')
		result_x_fin = (((2*np.pi)/(const.c))*result_x).to('erg cm-3 Hz-1')
		return result_x_fin, error_x_fin



	#########################################################################################################################################
	### Functions related to the IR torus transformation. We will perform the integration in cos(alpha), the incident angle from the disk ###
	#########################################################################################################################################

	# Define the normalization of the blackbody: we will normalize using the energy density for R_diss<R_IR and the luminosity for R_IR>R_diss

	def normalization_IR(self,R_diss):
		""" normalization of the IR blackbody: we will normalize using the energy density for R_diss<R_IR and the luminosity for R_IR>R_diss"""
		if (R_diss.to('cm')).value < (self.R_IR.to('cm')).value:
			u1 = ((self.f_IR*self.L_disk)/(4*np.pi*self.R_IR**2*const.c)).to('erg cm-3')
			u2 = ((4*const.sigma_sb*self.T_IR_rest**4)/(const.c)).to('erg cm-3')
			norm = u1/u2    
		else:
			norm = (self.f_IR*self.L_disk)/((4*np.pi*(self.R_IR**2)*const.sigma_sb*self.T_IR_rest**4).to('erg s-1'))
		return norm

	def final_fun_IR(self,cos_alpha,freq,R_diss):
	    mu = ext_rad.angle_inv(self.R_IR,cos_alpha,R_diss)
	    d = ext_rad.doppler(self.gamma,mu)
	    bb = ext_rad.normalization_IR(self,R_diss)*ext_rad.bb_nu(freq, self.T_IR_rest*d**-1)*cos_alpha
	    diff = (self.R_IR**2*(2*cos_alpha*(R_diss**2+self.R_IR**2)-2*self.R_IR*R_diss*(cos_alpha**2+1)))/(2*(2*self.R_IR*R_diss*cos_alpha-self.R_IR**2-R_diss**2)**2*np.sqrt((-2*self.R_IR*R_diss*cos_alpha+(R_diss**2)+(self.R_IR**2)*(cos_alpha**2))/(-2*self.R_IR*R_diss*cos_alpha+R_diss**2+(self.R_IR**2))))
	    prod = (d**2)*bb
	    return (prod.to('erg s-1 sr-1 cm-2 Hz-1'))*np.abs(diff)

	def u_IR(self,freq, R_diss):
		result = [integrate.quad(lambda x: ext_rad.final_fun_IR(self,x,i,R_diss).value,0, 1)[0] for i in freq]*u.erg*u.s**-1*u.cm**-2*u.Hz**-1
		error = [integrate.quad(lambda x: ext_rad.final_fun_IR(self,x,i,R_diss).value,0, 1)[1] for i in freq]*u.erg*u.s**-1*u.cm**-2*u.Hz**-1
		result_fin = (((2*np.pi)/(const.c))*result).to('erg cm-3 Hz-1')
		error_fin = (((2*np.pi)/(const.c))*error).to('erg cm-3 Hz-1')
		return result_fin, error_fin



	#########################################################################################################################################
	###                         Merge it all in one final function that returns the transformed three components                          ###
	#########################################################################################################################################

	# Note that for the disk and corona we have chosen the integration in logR
	def transform_external_fields(self,freq, R_diss):
		""" For a given distance from the black hole and frequency grid, returns the transformed fields from the disk and corona, the BLR and the IR torus.
		Parameters
		----------
		.self = Object of the class ext_rad that contains all the necessary parameters
		
		freq = array with astropy.units of frequency
	
		R_diss = distance from blackhole in cm
		"""

		if R_diss.to(u.cm) < 3*self.R_s.to(u.cm):
			print('Error: R_diss has to be greater than 3 times the Schwarzschild radius')
			u_disk_and_corona = np.nan
			u_BLR = np.nan
			u_IR = np.nan
		else:
			u_disk = ext_rad.u_d_logR(self,freq, R_diss)[0]
			u_corona = ext_rad.u_x_logR(self, freq, R_diss)[0]
			u_disk_and_corona = ext_rad.join_disk_corona(u_disk, u_corona, freq)

			u_BLR_disk = ext_rad.u_BLR_bb(self,freq, R_diss)[0]
			u_BLR_corona = ext_rad.u_BLR_x(self,freq, R_diss)[0]
			u_BLR = ext_rad.join_disk_corona(u_BLR_disk, u_BLR_corona, freq)

			u_IR = ext_rad.u_IR(self,freq, R_diss)[0]
		return u_disk_and_corona, u_BLR, u_IR

