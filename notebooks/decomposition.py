"""
NAME: decomposition

PURPOSE: Class to decompose bulge and disk and to compare them to best fit galpy potentials.

HISTORY:  22-10-2018 - Written - Milanov (ESO)
"""

import sys
sys.path.append("..")
from areposnap.gadget import gadget_readsnap
from areposnap.gadget_subfind import load_subfind

from auriga_basics import *

from galpy.potential import MiyamotoNagaiPotential, NFWPotential, HernquistPotential
from galpy.potential.plotRotcurve import vcirc
from galpy.util import bovy_conversion
import numpy as np

from scipy import stats

from astropy import units as u


from matplotlib import pyplot as plt




class decomposition():

	def __init__(self, machine = 'virgo', snapnr = 127, use_masses = False, use_n = True, galpyinputfile = None, galpyinputdata = None, have_galpy_potential = True):
		self.machine = machine
		self.snapnr = snapnr

		if self.machine == 'magny':
			self.filedir = "/home/extmilan/masterthesis/files/"
			self.basedir = "/hits/universe/GigaGalaxy/level4_MHD/"
			self.plotdir = "/home/extmilan/masterthesis/plots/"
		elif self.machine == 'mac':
			self.filedir = "/Users/smilanov/Documents/masterthesis/auriga_files/files/"
			self.basedir = "/Users/smilanov/Desktop/Auriga/level4/"
			self.plotdir = "/Users/smilanov/Documents/masterthesis/auriga_files/plots/"
		elif self.machine == 'virgo': 
			self.basedir = "/virgo/simulations/Auriga/level4_MHD/"
			self.filedir = "/u/milas/masterthesis/masterproject/files"
			self.plotdir = "/u/milas/masterthesis/masterproject/plots/"
		else:
			raise NotADirectoryError
            
		print('Load snapshot.')
		self._load_snapshot(snapnr)
		print("Carry out decomposition.")
		self.disk_IDs, self.spheroid_IDs = self._decomp()
		print('Calculate disk indices.')
		self._get_disk_indices()
		print('Calculate spheroid indices.')
		self._get_spheroid_indices()
		print("Calculate halo indices.")
		self._get_halo_indices()      
		print('Load positions and masses of simulation data.')
		self._get_disk_positions()
		self._get_spher_positions()
		self._get_halo_positions()
		self._get_disk_masses()
		self._get_spher_masses()
		self._get_halo_masses()
		self._get_circ_stars_pos_vel()
		if have_galpy_potential == True: 
			print('Import galpy parameters.')
			self._get_galpy_parameters(galpyinputfile, galpyinputdata)
			print("Setup galpy potential.")
			self._setup_galpy_potential(use_masses, use_n)
		else: print('Galpy potentials not initialized.')
		print('Set plot options.')
		self._set_plot_colors()


	def _get_galpy_parameters(self, inputfile, inputdata):
		if not isinstance(inputfile, str) and not isinstance(inputdata, np.ndarray):
			self.a_MND_kpc   = 2.96507719743
			self.b_MND_kpc   = 1.63627757204
			self.a_HB_kpc	 = 1.71545528287
			self.a_NFWH_kpc  = 26.0152749345
			self.v0_tot_kms  = 220.724646624
			self.v0_MND_kms  = 105.005287928
			self.v0_HB_kms   = 111.241910552
			self.v0_NFWH_kms = 159.117869741
			self.R0_kpc		 = 8.02852213383
			self.n_MND		 = self.v0_MND_kms**2  / self.v0_tot_kms**2
			self.n_HB		 = self.v0_HB_kms**2   / self.v0_tot_kms**2
			self.n_NFWH		 = self.v0_NFWH_kms**2 / self.v0_tot_kms**2
		elif isinstance(inputdata, np.ndarray):
			self.a_MND_kpc   = inputdata[0]
			self.b_MND_kpc   = inputdata[1]
			self.a_HB_kpc	 = inputdata[2]
			self.a_NFWH_kpc  = inputdata[3]
			self.v0_tot_kms  = inputdata[4]
			self.v0_MND_kms  = inputdata[5]
			self.v0_HB_kms   = inputdata[6]
			self.v0_NFWH_kms = inputdata[7]
			self.R0_kpc		 = inputdata[8]
			self.n_MND		 = self.v0_MND_kms**2  / self.v0_tot_kms**2
			self.n_HB		 = self.v0_HB_kms**2   / self.v0_tot_kms**2
			self.n_NFWH		 = self.v0_NFWH_kms**2 / self.v0_tot_kms**2
		elif isinstance(inputfile, str):
			data = np.loadtxt(self.filedir + inputfile)
			self.a_MND_kpc   = data[0]
			self.b_MND_kpc   = data[1]
			self.a_HB_kpc	 = data[2]
			self.a_NFWH_kpc  = data[3]
			self.v0_tot_kms  = data[4]
			self.v0_MND_kms  = data[5]
			self.v0_HB_kms   = data[6]
			self.v0_NFWH_kms = data[7]
			self.R0_kpc		 = data[8]
			self.n_MND		 = self.v0_MND_kms**2  / self.v0_tot_kms**2
			self.n_HB		 = self.v0_HB_kms**2   / self.v0_tot_kms**2
			self.n_NFWH		 = self.v0_NFWH_kms**2 / self.v0_tot_kms**2

	def _load_snapshot(self, snapnr, halo_number = 24):

			
		#### path = /hits/universe/GigaGalaxy/level4_MHD/halo_24/output/*
		level = 4

		for halo_number in [halo_number]:  # range(1, 31):
			halodir = self.basedir+"halo_{0}/".format(halo_number)
			snappath = halodir+"output/"

			print("level   : {0}".format(level))
			print("halo	: {0}".format(halo_number))
			print("snapnr  : {0}".format(snapnr))
			print("basedir : {0}".format(self.basedir))
			print("halodir : {0}".format(halodir))
			print("snappath: {0}\n".format(snappath))
			self.s, self.sf = eat_snap_and_fof(level, halo_number, snapnr, snappath, loadonlytype=[1,2,3,4], 
				haloid=0, galradfac=0.1, verbose=True) 

			# Clean negative and zero values of gmet to avoid RuntimeErrors
			# later on (e.g. dividing by zero)
			self.s.data['gmet'] = np.maximum( self.s.data['gmet'], 1e-40 )

##### Set up galpy potential #####
	def _setup_galpy_potential(self, use_masses = False, use_n = True, whole_potential = True):
		#test input:
		if whole_potential == True:
			if (self.a_MND_kpc <= 0.) or (self.b_MND_kpc <= 0.) or (self.a_NFWH_kpc <= 0.) or (self.a_HB_kpc <= 0.) \
			   or (self.n_MND <= 0.) or (self.n_NFWH <= 0.) or (self.n_HB <= 0.) or (self.n_MND >= 1.) or (self.n_NFWH >= 1.) or (self.n_HB >= 1.):
				raise ValueError('Error in setup_galpy_potential: '+\
								 'The input parameters for the scaling profiles do not correspond to a physical potential.')
			if np.fabs(self.n_MND + self.n_NFWH + self.n_HB - 1.) > 1e-7:
				raise ValueError('Error in setup_galpy_potential: '+\
								 'The sum of the normalization does not add up to 1.')
			
		#trafo to galpy units:
		a_MND  = self.a_MND_kpc  / self.R0_kpc
		b_MND  = self.b_MND_kpc  / self.R0_kpc
		a_NFWH = self.a_NFWH_kpc / self.R0_kpc
		a_HB   = self.a_HB_kpc   / self.R0_kpc
        
		if use_masses == True:
			nfw_mass = np.sum(self.s.mass[((self.s.type == 1) + (self.s.type == 2) + (self.s.type == 3)) * (self.s.halo == 0) * (self.s.subhalo == 0)])* 1e10 * u.Msun #* (self.s.r()<=self.s.galrad)
			hb_mass = 10**10*np.sum(self.s.mass[self.i_spher][self.s.r()[self.i_spher] <= self.s.galrad])*u.Msun
			#setup potential:
			self.disk = MiyamotoNagaiPotential(amp=10**10*np.sum(self.disk_masses_10msun)*u.Msun, a=self.a_MND_kpc*u.kpc, b=self.b_MND_kpc*u.kpc,)
			self.halo = NFWPotential(amp=10**10*np.sum(self.s.mass[self.i_halo][self.i_r_inhalo])*u.Msun, a = self.a_NFWH_kpc*u.kpc)
			self.bulge = HernquistPotential(amp=10**10*np.sum(self.s.mass[self.i_spher][self.i_r_inspher])*u.Msun, a = self.a_HB_kpc*u.kpc) 
            
		elif use_n == True:
			self.disk = MiyamotoNagaiPotential(a=a_MND, b=b_MND,normalize = self.n_MND)
			self.halo = NFWPotential(a = a_NFWH, normalize=self.n_NFWH)
			self.bulge = HernquistPotential(a = a_HB, normalize = self.n_HB)         
		else: print('Please use either n or masses.')        
		self.pot = [self.disk,self.halo,self.bulge]
		#return [disk,halo,bulge], disk, halo, bulge


	def _decomp(self, circ_val = 0.7, plotter = False, savefig = False, include_zmax = False, zmax = 0.0005, Gcosmo = 43.0071):
		ID = self.s.id
		# get number of particle 
		na = self.s.nparticlesall

		# get mass and age of stars 
		mass = self.s.data['mass'].astype('float64')
		st = na[:4].sum(); en = st+na[4]
		age = np.zeros( self.s.npartall )
		# only stars will be given an age, for other particles: age = 0
		age[st:en] = self.s.data['age']

		# create masks for all particles / stars within given radius
		iall, = np.where( (self.s.r() < self.s.galrad) & (self.s.r() > 0.) )
		istars, = np.where( (self.s.r() < self.s.galrad) & (self.s.r() > 0.) & (self.s.type == 4) & (age > 0.) )

		# calculate radius of all particles within galrad, sort them, \
		# get their mass and calculate their cumulative mass sorted by their distance
		nstars = len( istars )
		nall   = len( iall )
		rr = np.sqrt( (self.s.pos[iall,:]**2).sum(axis=1) )
		msort = rr.argsort()
		mass_all = mass[iall]
		msum = np.zeros( nall )
		msum[msort[:]] = np.cumsum(mass_all[msort[:]])

		# get position, velocity, mass, type, potential, radius and age of all particles within galrad
		pos  = self.s.pos[iall,:].astype( 'float64' )
		vel = self.s.vel[iall,:].astype( 'float64' )
		mass = self.s.data['mass'][iall].astype( 'float64' )
		ptype = self.s.data['type'][iall]
		pot = self.s.data['pot'][iall].astype('float64')
		radius = np.sqrt( (self.s.pos[iall,:]**2).sum(axis=1) )
		Radius = np.sqrt((self.s.pos[iall,2]**2 + self.s.pos[iall,1]**2))
		age = age[iall]

		# create zero arrays with the length = number of stars for values to be calculated
		eps   = np.zeros( nstars )
		eps2  = np.zeros( nstars )
		smass = np.zeros( nstars )
		cosalpha = np.zeros( nstars )
		jcmax = np.zeros( nstars )
		spec_energy = np.zeros( nstars )
		star_age = np.zeros( nstars )

		# computing stellar properties
		print('computing star properties')

		# another mask for star selection
		nn, = np.where((ptype[:] == 4) & (age[:] > 0.)) 

		# calculate specific angular momentum (w/o considering mass)
		j  = np.cross( pos[nn,:], vel[nn,:] ) #!!!! Check if x, y, z position is the right way
		# calculate circular orbit angular momentum
		jc = radius[nn] * np.sqrt( Gcosmo * msum[nn] / radius[nn] )
		# get z-component of specific angular momentum !! CHECK AGAIN XYZ 
		jz = j[:,0]

		# calculate specific energy
		spec_energy[:] = 0.5 * (vel[nn,:]**2).sum(axis=1) + pot[nn]
		# circularity parameter eps = j_z / j_circ
		eps[:] = jz / jc
		eps2[:] = jz
		smass[:] = mass[nn]
		# calculate normed Jz??
		cosalpha[:] = jz / np.sqrt( (j[:]**2).sum(axis=1) )
		# calculate age from cosmology
		star_age[:] = self.s.cosmology_get_lookback_time_from_a( age[nn], is_flat=True )

		print('computing histograms')

		# sort particle by specific energy
		iensort = np.argsort(spec_energy)
		eps = eps[iensort]
		eps2 = eps2[iensort]
		spec_energy = spec_energy[iensort]
		smass = smass[iensort]
		cosalpha = cosalpha[iensort]
		star_age = star_age[iensort]

		# find maximum allowed J_circ for each energy for each star 
		#(look at 100 stars around selected star and find the maximum jz within them)
		for nn in range( nstars ):
			nn0 = nn - 50
			nn1 = nn + 50

			if nn0 < 0:
				nn1 += -nn0
				nn0 = 0
			if nn1 >= nstars:
				nn0 -= ( nn1 - (nstars - 1) )
				nn1 = nstars - 1

			jcmax[nn] = np.max( eps2[nn0:nn1] )
		nn, = np.where((ptype[:] == 4) & (age[:] > 0.)) 

		# divide star mass by mass of all stars to use as weight in histogram
		smass /= smass.sum()
		# calculate eps = jz/j_z_max 
		eps2[:] /= jcmax[:]

		Radius_Mpc = self.s.r()[istars][iensort]
		
		if include_zmax == True:
			idisk, = np.where((eps2 >= circ_val) & (abs(self.s.pos[:,0][istars][iensort]) < zmax))
			disk_ID = ID[istars][iensort][idisk]
			ispheroid = np.where(((eps2 < circ_val) + ((eps2 >= circ_val) & (abs(self.s.pos[:,0][istars][iensort]) > zmax))))
			spheroid_ID = ID[istars][iensort][ispheroid]

		else:
			idisk, = np.where((eps2 >= circ_val)) 
			disk_ID = ID[istars][iensort][idisk]
			ispheroid = np.where(eps2 < circ_val)
			spheroid_ID = ID[istars][iensort][ispheroid]
		
		if plotter == True:
			fig, ax = plt.subplots(1, 1)
			
			ydatatot, edges = np.histogram( eps2, bins=100, weights=smass, range=[-1.7,1.7] )
			xdatatot = 0.5 * (edges[1:] + edges[:-1])
			xdata = xdatatot

			ydatad, edgesd = np.histogram( eps2[idisk], bins=100, weights=smass[idisk], range=[-1.7,1.7] )
			ydatas, edgess = np.histogram( eps2[ispheroid], bins=100, weights=smass[ispheroid], range=[-1.7,1.7] )

			ax.fill( xdata, ydatad, fc='b', alpha=0.5, fill=True, lw=0, label='disk' )
			ax.fill( xdata, ydatas, fc='g', alpha=0.5, fill=True, lw=0, label='spheroid' )
			
			ax.plot( xdatatot, ydatatot, 'k', label='total' )
			ax.legend()
			ax.set_xlabel('$\epsilon$')
			if savefig == True:
				fig.savefig(self.plotdir + 'potential/decomposition_snap_{}.png'.format(self.snapnr), format = 'png', dpi = 300, bbox_to_inches = 'thight')
			plt.show()
			
		return(disk_ID, spheroid_ID)    

	def _get_disk_indices(self):
		self.i_disk = np.isin(self.s.id, self.disk_IDs)

	def _get_spheroid_indices(self):
		self.i_spher = np.isin(self.s.id, self.spheroid_IDs)
    
	def _get_halo_indices(self):
		self.i_halo = np.where(((self.s.type ==1  ) +  (self.s.type == 2) + (self.s.type == 3))* (self.s.halo ==0)* (self.s.subhalo == 0), True, False)# 

	def _get_disk_positions(self):
		self.r_disk_kpc = 1000. * self.s.r()[self.i_disk]
		self.x_disk_kpc = 1000. * self.s.pos[:, 2][self.i_disk]
		self.y_disk_kpc = 1000. * self.s.pos[:, 1][self.i_disk]
		self.z_disk_kpc = 1000. * self.s.pos[:, 0][self.i_disk]
		self.R_disk_kpc = np.sqrt(self.x_disk_kpc**2 + self.y_disk_kpc**2)


		# keep it within galrad
		self.i_r_indisk = self.r_disk_kpc <= (1000. * self.s.galrad)
	 
		# keep it within zmin - zmax range
		self.i_z_indisk = (self.z_disk_kpc >= -5.) * (self.z_disk_kpc <= 5.)

	def _get_spher_positions(self):
		self.r_spher_kpc = 1000. * self.s.r()[self.i_spher]
		self.x_spher_kpc = 1000. * self.s.pos[:, 2][self.i_spher]
		self.y_spher_kpc = 1000. * self.s.pos[:, 1][self.i_spher]
		self.z_spher_kpc = 1000. * self.s.pos[:, 0][self.i_spher]
		self.R_spher_kpc = np.sqrt(self.x_spher_kpc**2 + self.y_spher_kpc**2)


		# keep it within galrad
		self.i_r_inspher = self.r_spher_kpc <= (1000. * self.s.galrad)
		self.i_z_inspher = (self.z_spher_kpc >= -5.) * (self.z_spher_kpc <= 5.)
        
	def _get_halo_positions(self):
		self.r_halo_kpc = 1000. * self.s.r()[self.i_halo]
		self.x_halo_kpc = 1000. * self.s.pos[:, 2][self.i_halo]
		self.y_halo_kpc = 1000. * self.s.pos[:, 1][self.i_halo]
		self.z_halo_kpc = 1000. * self.s.pos[:, 0][self.i_halo]
		self.R_halo_kpc = np.sqrt(self.x_halo_kpc**2 + self.y_halo_kpc**2)


		# keep it within galrad
		self.i_r_inhalo = self.r_halo_kpc <= (1000. * self.s.galrad)
		self.i_z_inhalo = (self.z_halo_kpc >= -5.) * (self.z_halo_kpc <= 5.)
        
	def _get_circ_stars_pos_vel(self):
		self.ID_disk_circ, self.ID_spher_circ = self._decomp(circ_val = 0.98, include_zmax = True, zmax = 0.005)
		self.i_disk_circ= np.isin(self.s.id, self.ID_disk_circ)
		(self.R_circ_stars_kpc, self.phi_circ_stars_, self.z_circ_stars_kpc), (self.vR_circ_stars_kms, self.vphi_circ_stars_kms, self.vz_circ_stars_kms) = get_cylindrical_vectors(self.s, self.sf, self.i_disk_circ)
		self.r_circ_stars_kpc = 1000. * self.s.r()[self.i_disk_circ]
		self.i_r_circ_stars_in = self.r_circ_stars_kpc <= (1000. * self.s.galrad)

		
	def _get_disk_masses(self):
		self.disk_masses_10msun = self.s.mass[self.i_disk][self.i_r_indisk * self.i_z_indisk]
        
	def _get_spher_masses(self):
		self.spher_masses_10msun = self.s.mass[self.i_spher][self.i_r_inspher* self.i_z_inspher]

	def _get_halo_masses(self):
		self.halo_masses_10msun = self.s.mass[self.i_halo][self.i_r_inhalo* self.i_z_inhalo]
        
	def _set_plot_colors(self):
		self.disk_color = 'blue'
		self.halo_color = 'orange'
		self.spher_color = 'green'
		self.tot_color = 'red'
##### Circular velocities #####
        
	def circvel_data(self, N = 25):
		v_mean_kms, R_bin_edges, binnum = stats.binned_statistic(self.R_circ_stars_kpc[self.i_r_circ_stars_in], np.abs(self.vphi_circ_stars_kms[self.i_r_circ_stars_in]), bins = N)
		R_bins_kpc = R_bin_edges[:-1] + 1./2. * (R_bin_edges[1:] - R_bin_edges[:-1])
		return(v_mean_kms, R_bins_kpc)

	def circvel_galpy(self, R_bins_kpc, N = 25, use_masses = False, use_n = True):
		vcirc_disk_bestfit_kms  = np.zeros(N)
		vcirc_spher_bestfit_kms = np.zeros(N)
		vcirc_halo_bestfit_kms  = np.zeros(N)
		vcirc_tot_bestfit_kms   = np.zeros(N)
		if use_masses == True:
			for i, item in enumerate(R_bins_kpc):
				vcirc_disk_bestfit_kms[i]  = self.disk.vcirc(item*u.kpc)		
				vcirc_halo_bestfit_kms[i]  = self.halo.vcirc(item*u.kpc)	
				vcirc_spher_bestfit_kms[i] = self.bulge.vcirc(item*u.kpc)	
				vcirc_tot_bestfit_kms[i]   = vcirc(self.pot, item*u.kpc)
		elif use_n == True:
			for i, item in enumerate(R_bins_kpc):
				item_galpy = item / self.R0_kpc
				vcirc_disk_bestfit_kms[i]  = self.disk.vcirc(item_galpy)*self.v0_tot_kms		
				vcirc_halo_bestfit_kms[i]  = self.halo.vcirc(item_galpy)*self.v0_tot_kms	
				vcirc_spher_bestfit_kms[i] = self.bulge.vcirc(item_galpy)*self.v0_tot_kms	
				vcirc_tot_bestfit_kms[i]   = vcirc(self.pot, item_galpy)*self.v0_tot_kms
		return(vcirc_tot_bestfit_kms, vcirc_disk_bestfit_kms, vcirc_spher_bestfit_kms, vcirc_halo_bestfit_kms)

	def plot_circvel(self, N = 25, safefigure = True, use_masses = False, use_n = True):
		v_mean_kms, R_bins_kpc = self.circvel_data(N)
		vcirc_tot_bestfit_kms, vcirc_disk_bestfit_kms, vcirc_spher_bestfit_kms, vcirc_halo_bestfit_kms = self.circvel_galpy(R_bins_kpc, N, use_masses, use_n)

		fig,ax = plt.subplots(figsize = (8,8))
		ax.spines['top'].set_linewidth(1.5)
		ax.spines['bottom'].set_linewidth(1.5)
		ax.spines['left'].set_linewidth(1.5)
		ax.spines['right'].set_linewidth(1.5)
		ax.xaxis.set_tick_params(width=1.5)
		ax.yaxis.set_tick_params(width=1.5)
		ax.plot(R_bins_kpc, v_mean_kms, 'k.', label = 'data')
		ax.plot(R_bins_kpc, vcirc_tot_bestfit_kms, color = self.tot_color, linestyle = '-', label = 'fit total')
		ax.plot(R_bins_kpc, vcirc_disk_bestfit_kms, color = self.disk_color, linestyle = '--', label = 'fit disk')
		ax.plot(R_bins_kpc, vcirc_spher_bestfit_kms, color = self.spher_color, linestyle = '-.', label = 'fit spheroid')
		ax.plot(R_bins_kpc, vcirc_halo_bestfit_kms, color = self.halo_color, linestyle = ':', label = 'fit halo')
		ax.set_ylabel('circular velocity [km s$^{-1}$]')
		ax.set_xlabel('R [kpc]')
		ax.legend(loc = 4)
		fig.tight_layout()
		if safefigure == True:
			fig.savefig(self.plotdir + 'circ_vel_fit_data_test.png', dpi = 300, format = 'png' )
		plt.show()

##### Surface densities #####
        
	def surfdens_disk_data(self, N = 25):
		mass_hist, R_bin_edges = np.histogram(self.R_disk_kpc[self.i_r_indisk * self.i_z_indisk], weights = self.disk_masses_10msun, bins = N)
		area = np.pi * (R_bin_edges[1:]**2 - R_bin_edges[:-1]**2)
		rho_Msun_pc2 = 1e4 * mass_hist / area
		R_mean_kpc = R_bin_edges[:-1] + 1./2. * (R_bin_edges[1:] - R_bin_edges[:-1])
		return(rho_Msun_pc2, R_mean_kpc)

	def surfdens_disk_galpy(self, R_bins_kpc, z_extend_kpc = 5., use_masses = False, use_n = True):
		surfdens_bestfit = np.zeros(len(R_bins_kpc))
		if use_masses == True:
			for i, item in enumerate(R_bins_kpc):
				surfdens_bestfit[i] = self.disk.surfdens(item * u.kpc, z_extend_kpc * u.kpc) 
		elif use_n == True:
			for i, item in enumerate(R_bins_kpc):
				item_galpy = item / self.R0_kpc
				z_extend_galpy = z_extend_kpc / self.R0_kpc
				surfdens_bestfit[i] = self.disk.surfdens(item_galpy, z_extend_galpy) * bovy_conversion.surfdens_in_msolpc2(self.v0_tot_kms, self.R0_kpc)
		return(surfdens_bestfit)
    
	def surfdens_spher_data(self, N = 25):
		mass_hist, R_bin_edges = np.histogram(self.R_spher_kpc[self.i_r_inspher* self.i_z_inspher], weights = self.spher_masses_10msun, bins = N)
		area = np.pi * (R_bin_edges[1:]**2 - R_bin_edges[:-1]**2)
		rho_Msun_pc2 = 1e4 * mass_hist / area
		R_mean_kpc = R_bin_edges[:-1] + 1./2. * (R_bin_edges[1:] - R_bin_edges[:-1])
		return(rho_Msun_pc2, R_mean_kpc)

	def surfdens_spher_galpy(self, R_bins_kpc, z_extend_kpc = 5., use_masses = False, use_n = True):
		surfdens_bestfit = np.zeros(len(R_bins_kpc))
		if use_masses == True:
			for i, item in enumerate(R_bins_kpc):
				surfdens_bestfit[i] = self.bulge.surfdens(item * u.kpc, z_extend_kpc * u.kpc) #### check how surface density of bulge is calculated?
		elif use_n == True:
			for i, item in enumerate(R_bins_kpc):
				item_galpy = item / self.R0_kpc
				z_extend_galpy = z_extend_kpc / self.R0_kpc
				surfdens_bestfit[i] = self.bulge.surfdens(item_galpy, z_extend_galpy) * bovy_conversion.surfdens_in_msolpc2(self.v0_tot_kms, self.R0_kpc)
		return(surfdens_bestfit)
    
	def surfdens_halo_data(self, N = 25):
		mass_hist, R_bin_edges = np.histogram(self.R_halo_kpc[self.i_r_inhalo* self.i_z_inhalo], weights = self.halo_masses_10msun, bins = N)
		area = np.pi * (R_bin_edges[1:]**2 - R_bin_edges[:-1]**2)
		rho_Msun_pc2 = 1e4 * mass_hist / area
		R_mean_kpc = R_bin_edges[:-1] + 1./2. * (R_bin_edges[1:] - R_bin_edges[:-1])
		return(rho_Msun_pc2, R_mean_kpc)

	def surfdens_halo_galpy(self, R_bins_kpc, z_extend_kpc = 5., use_masses = False, use_n = True):
		surfdens_bestfit = np.zeros(len(R_bins_kpc))
		if use_masses == True:
			for i, item in enumerate(R_bins_kpc):
				surfdens_bestfit[i] = self.halo.surfdens(item * u.kpc, z_extend_kpc * u.kpc) #### check how surface density of bulge is calculated?
		elif use_n == True:
			for i, item in enumerate(R_bins_kpc):
				item_galpy = item / self.R0_kpc
				z_extend_galpy = z_extend_kpc / self.R0_kpc
				surfdens_bestfit[i] = self.halo.surfdens(item_galpy, z_extend_galpy) * bovy_conversion.surfdens_in_msolpc2(self.v0_tot_kms, self.R0_kpc)
		return(surfdens_bestfit)
        
	def plot_surfdens_disk(self, N = 25, safefigure = True, use_masses = False, use_n = True):
		surfdens_data_Msun_pc2_data, R_bins_kpc = self.surfdens_disk_data(N)
		surfdens_bestfit_Msun_pc2 = self.surfdens_disk_galpy(R_bins_kpc, use_masses = use_masses, use_n = use_n)
		fig,ax = plt.subplots(figsize = (8,8))
		ax.spines['top'].set_linewidth(1.5)
		ax.spines['bottom'].set_linewidth(1.5)
		ax.spines['left'].set_linewidth(1.5)
		ax.spines['right'].set_linewidth(1.5)
		ax.xaxis.set_tick_params(width=1.5)
		ax.yaxis.set_tick_params(width=1.5)
		ax.semilogy(R_bins_kpc, surfdens_data_Msun_pc2_data, 'k.', label = 'data')
		ax.semilogy(R_bins_kpc, surfdens_bestfit_Msun_pc2, color = self.disk_color, linestyle = '-', label = 'best fit')
		ax.set_ylabel('disk surface density [M$_\odot$  pc$^{-2}$]')
		ax.set_xlabel('R [kpc]')
		ax.legend()
		fig.tight_layout()
		if safefigure == True:
			fig.savefig(self.plotdir + 'surface_dens_disk_fit_data.png', dpi = 300, format = 'png' )
		plt.show()

	def plot_surfdens_spher(self, N = 25, safefigure = True, use_masses = False, use_n = True):
		surfdens_spher_data_Msun_pc2_data, r_bins_kpc = self.surfdens_spher_data(N)
		surfdens_spher_bestfit_Msun_pc2 = self.surfdens_spher_galpy(r_bins_kpc, use_masses = use_masses, use_n = use_n)
		fig,ax = plt.subplots(figsize = (8,8))
		ax.spines['top'].set_linewidth(1.5)
		ax.spines['bottom'].set_linewidth(1.5)
		ax.spines['left'].set_linewidth(1.5)
		ax.spines['right'].set_linewidth(1.5)
		ax.xaxis.set_tick_params(width=1.5)
		ax.yaxis.set_tick_params(width=1.5)
		ax.semilogy(r_bins_kpc, surfdens_spher_data_Msun_pc2_data, 'k.', label = 'data')
		ax.semilogy(r_bins_kpc, surfdens_spher_bestfit_Msun_pc2, color = self.spher_color, linestyle = '-', label = 'best fit')
		ax.set_ylabel('bulge surface density [M$_\odot$  pc$^{-2}$]')
		ax.set_xlabel('r [kpc]')
		ax.legend()
		fig.tight_layout()
		if safefigure == True:
			fig.savefig(self.plotdir + 'surface_dens_spher_fit_data.png', dpi = 300, format = 'png' )
		plt.show()

	def plot_surfdens_halo(self, N = 25, safefigure = True, use_masses = False, use_n = True):
		surfdens_halo_data_Msun_pc2_data, r_bins_kpc = self.surfdens_halo_data(N)
		surfdens_halo_bestfit_Msun_pc2 = self.surfdens_halo_galpy(r_bins_kpc, use_masses = use_masses, use_n = use_n)
		fig,ax = plt.subplots(figsize = (8,8))
		ax.spines['top'].set_linewidth(1.5)
		ax.spines['bottom'].set_linewidth(1.5)
		ax.spines['left'].set_linewidth(1.5)
		ax.spines['right'].set_linewidth(1.5)
		ax.xaxis.set_tick_params(width=1.5)
		ax.yaxis.set_tick_params(width=1.5)
		ax.semilogy(r_bins_kpc, surfdens_halo_data_Msun_pc2_data, 'k.', label = 'data')
		ax.semilogy(r_bins_kpc, surfdens_halo_bestfit_Msun_pc2, color = self.halo_color, linestyle ='-', label = 'best fit')
		ax.set_ylabel('halo surface density [M$_\odot$  pc$^{-2}$]')
		ax.set_xlabel('r [kpc]')
		ax.legend()
		fig.tight_layout()
		if safefigure == True:
			fig.savefig(self.plotdir + 'surface_dens_halo_fit_data.png', dpi = 300, format = 'png' )
		plt.show()

##### Volume densities #####        

	def voldens_disk_1d_data(self, N = 25, z_kpc = None, R_kpc = None):
        ##### aufteilung funktioniert so nicht ####
		if z_kpc is not None:
			mass_hist, R_bin_edges = np.histogram(self.R_disk_kpc[self.i_r_indisk], weights = self.s.mass[self.i_disk][self.i_r_indisk], bins = N)
			volume = np.pi * (R_bin_edges[1:]**2 - R_bin_edges[:-1]**2) * z_kpc
			rho_Msun_pc3 = 10. * mass_hist / volume
			R_mean_kpc = R_bin_edges[:-1] + 1./2. * (R_bin_edges[1:] - R_bin_edges[:-1])
			return(rho_Msun_pc3, R_mean_kpc)
		elif R_kpc is not None:
			mass_hist, z_bin_edges = np.histogram(self.z_disk_kpc[self.i_r_in], weights = self.s.mass[self.i_disk][self.i_r_indisk], bins = N)
			z_mean_kpc = z_bin_edges[:-1] + 1./2. * (z_bin_edges[1:] - z_bin_edges[:-1])
			volume = np.pi * R_kpc**2 * z_mean_kpc
			rho_Msun_pc3 = 10. * mass_hist / volume
			return(rho_Msun_pc3, z_mean_kpc)
		else:print('Please specify z OR R [kpc].')
            
	def voldens_disk_2d_data(self, dR_kpc = 1., dz_kpc = 0.5):    
		indisk = self.i_r_indisk * self.i_z_indisk
		Rmin_kpc, Rmax_kpc = np.min(self.R_disk_kpc[indisk]), np.max(self.R_disk_kpc[indisk])
		zmin_kpc, zmax_kpc = np.min(self.z_disk_kpc[indisk]), np.max(self.z_disk_kpc[indisk])
    
		N_R = int((Rmax_kpc - Rmin_kpc) / dR_kpc)
		N_z = int((zmax_kpc - zmin_kpc) / dz_kpc)        
		Rbins, zbins = np.linspace(Rmin_kpc, Rmax_kpc, N_R), np.linspace(zmin_kpc, zmax_kpc, N_z)
		mbins, volbins = np.zeros((len(zbins), len(Rbins))), np.zeros((len(zbins), len(Rbins))) 
		for i in range(len(zbins)):
			for j in range(len(Rbins)):
				inbin = (Rbins[j] <= self.R_disk_kpc[indisk]) & (self.R_disk_kpc[indisk] < (Rbins[j] + dR_kpc)) & (zbins[i] <= self.z_disk_kpc[indisk]) & (self.z_disk_kpc[indisk] < (zbins[i] + dz_kpc))
				mbins[i,j] = np.sum(self.disk_masses_10msun[inbin])
				volbins[i,j] = np.pi * dz_kpc * (2. * Rbins[j] * dR_kpc + dR_kpc**2)
       
		rho = mbins / volbins
		return(rho, Rbins, zbins)
        
	def voldens_disk_1d_galpy(self, R_kpc, z_kpc, fix_z = True, fiz_R = False, use_masses = False, use_n = True):
		if fix_z == True:
			voldens_bestfit = np.zeros(len(R_kpc))
			if use_masses == True:
				for i, item in enumerate(R_kpc):
					voldens_bestfit[i] = self.disk.dens(item * u.kpc, z_kpc * u.kpc) 
			elif use_n == True:
				for i, item in enumerate(R_kpc):
					item_galpy = item / self.R0_kpc
					z_fix_galpy = z_kpc / self.R0_kpc
					voldens_bestfit[i] = self.disk.dens(item_galpy, z_fix_galpy) * bovy_conversion.dens_in_msolpc3(self.v0_tot_kms, self.R0_kpc)
                    
		elif fix_R == True:
			voldens_bestfit = np.zeros(len(z_kpc))
			if use_masses == True:
				for i, item in enumerate(z_kpc):
					voldens_bestfit[i] = self.disk.dens(R_kpc * u.kpc, item * u.kpc) 
			elif use_n == True:
				for i, item in enumerate(z_kpc):
					item_galpy = item / self.R0_kpc
					R_fix_galpy = R_kpc / self.R0_kpc
					voldens_bestfit[i] = self.disk.dens(R_fix_galpy, item_galpy) * bovy_conversion.dens_in_msolpc3(self.v0_tot_kms, self.R0_kpc)

		return(voldens_bestfit)
    
	def voldens_disk_2d_galpy(self, use_masses = True, use_n = False, dR_kpc = 1., dz_kpc = 0.5):
		indisk = self.i_r_indisk * self.i_z_indisk
		Rmin_kpc, Rmax_kpc = np.min(self.R_disk_kpc[indisk]), np.max(self.R_disk_kpc[indisk])
		zmin_kpc, zmax_kpc = np.min(self.z_disk_kpc[indisk]), np.max(self.z_disk_kpc[indisk])
    
		N_R = int((Rmax_kpc - Rmin_kpc) / dR_kpc)
		N_z = int((zmax_kpc - zmin_kpc) / dz_kpc)        
		Rbins, zbins = np.linspace(Rmin_kpc, Rmax_kpc, N_R), np.linspace(zmin_kpc, zmax_kpc, N_z)
		voldens_bestfit = np.zeros((len(zbins), len(Rbins)))
		for i, z_item in enumerate(zbins):
			for j, R_item in enumerate(Rbins):
				if use_masses == True:
					voldens_bestfit[i, j] = self.disk.dens(R_item * u.kpc, z_item * u.kpc) * bovy_conversion.dens_in_msolpc3(self.v0_tot_kms, self.R0_kpc)
				elif use_n == True:
					R_galpy = R_item / self.R0_kpc
					z_galpy = z_item / self.R0_kpc
					voldens_bestfit[i, j] = self.disk.dens(R_galpy, z_galpy) * bovy_conversion.dens_in_msolpc3(self.v0_tot_kms, self.R0_kpc)
		return(voldens_bestfit, Rbins, zbins)   
    
	def voldens_spher_data(self, N = 25):
		r_spher_kpc = 1000. * self.s.r()[self.i_spher]      
		mass_hist, r_bin_edges = np.histogram(r_spher_kpc, weights = self.s.mass[self.i_spher][self.i_r_inspher], bins = N)
		volume = 4./3. * np.pi * (r_bin_edges[1:]**3 - r_bin_edges[:-1]**3)
		rho_Msun_pc3 = 10. * mass_hist / volume
		r_mean_kpc = r_bin_edges[:-1] + 1./2. * (r_bin_edges[1:] - r_bin_edges[:-1])
		return(rho_Msun_pc3, r_mean_kpc)
    
	def voldens_spher_galpy(self, r_kpc, use_masses = False, use_n = True):
		voldens_bestfit = np.zeros(len(r_kpc))
		if use_masses == True:
			for i, item in enumerate(r_kpc):
				voldens_bestfit[i] = self.bulge.dens(item * u.kpc, 0. * u.kpc) #### check how density of bulge is calculated?
		elif use_n == True:
			for i, item in enumerate(r_kpc):
				item_galpy = item / self.R0_kpc
				voldens_bestfit[i] = self.bulge.dens(item_galpy, 0.) * bovy_conversion.dens_in_msolpc3(self.v0_tot_kms, self.R0_kpc)
		return(voldens_bestfit)
    
	def voldens_halo_data(self, N = 25):
		r_halo_kpc = 1000. * self.s.r()[self.i_halo][self.i_r_inhalo]      
		mass_hist, r_bin_edges = np.histogram(r_halo_kpc, weights = self.s.mass[self.i_halo][self.i_r_inhalo], bins = N)
		volume = 4./3. * np.pi * (r_bin_edges[1:]**3 - r_bin_edges[:-1]**3)
		rho_Msun_pc3 = 10. * mass_hist / volume
		r_mean_kpc = r_bin_edges[:-1] + 1./2. * (r_bin_edges[1:] - r_bin_edges[:-1])
		return(rho_Msun_pc3, r_mean_kpc)
    
	def voldens_halo_galpy(self, r_kpc, use_masses = False, use_n = True):
		voldens_bestfit = np.zeros(len(r_kpc))
		if use_masses == True:
			for i, item in enumerate(r_kpc):
				voldens_bestfit[i] = self.halo.dens(item * u.kpc, 0. * u.kpc) #### check how density of bulge is calculated?
		elif use_n == True:
			for i, item in enumerate(r_kpc):
				item_galpy = item / self.R0_kpc
				voldens_bestfit[i] = self.halo.dens(item_galpy, 0.) * bovy_conversion.dens_in_msolpc3(self.v0_tot_kms, self.R0_kpc)
		return(voldens_bestfit)

	def plot_voldens_disk(self, safefigure = True, N = 25, fix_z = True, fix_R = False, R_dat_kpc = 5.):
		if fix_z == True:
			z_dat_kpc = np.median(np.abs(self.z_disk_kpc))
			voldens_data_Msun_pc3_data, R_bins_kpc = self.voldens_disk_data(N, z_kpc = z_dat_kpc, R_kpc = self.R_disk_kpc)
			voldens_bestfit_Msun_pc3 = self.voldens_disk_galpy(R_bins_kpc, z_dat_kpc)
			filename = 'volume_dens_disk_fit_data_z_{}_kpc.png'.format(z_dat_kpc)
		elif fix_R == True:
			voldens_data_Msun_pc3_data, R_bins_kpc = self.voldens_disk_data(N, z_kpc = self.z_disk_kpc, R_kpc = R_dat_kpc)
			voldens_bestfit_Msun_pc3 = self.voldens_disk_galpy(R_dat_kpc, self.z_disk_kpc)
			filename = 'volume_dens_disk_fit_data_R_{}_kpc.png'.format(R_dat_kpc)
		fig,ax = plt.subplots(figsize = (8,8))
		ax.spines['top'].set_linewidth(1.5)
		ax.spines['bottom'].set_linewidth(1.5)
		ax.spines['left'].set_linewidth(1.5)
		ax.spines['right'].set_linewidth(1.5)
		ax.xaxis.set_tick_params(width=1.5)
		ax.yaxis.set_tick_params(width=1.5)
		ax.plot(R_bins_kpc, voldens_data_Msun_pc3_data, 'k.', label = 'data')
		ax.plot(R_bins_kpc, voldens_bestfit_Msun_pc3, color = self.disk_color, linestyle = '-', label = 'best fit')
		ax.set_ylabel('disk volume density [M$_\odot$  pc$^{-3}$]')
		ax.set_xlabel('R [kpc]')
		ax.legend()
		fig.tight_layout()
		if safefigure == True:
			fig.savefig(self.plotdir + filename, dpi = 300, format = 'png' )
		plt.show()
        
	def plot_voldens_spher(self, safefigure = True, N = 25):
		voldens_spher_data_Msun_pc3_data, r_bins_kpc = self.voldens_spher_data(N)
		voldens_spher_bestfit_Msun_pc3 = self.voldens_spher_galpy(r_bins_kpc)
		fig,ax = plt.subplots(figsize = (8,8))
		ax.spines['top'].set_linewidth(1.5)
		ax.spines['bottom'].set_linewidth(1.5)
		ax.spines['left'].set_linewidth(1.5)
		ax.spines['right'].set_linewidth(1.5)
		ax.xaxis.set_tick_params(width=1.5)
		ax.yaxis.set_tick_params(width=1.5)

		ax.semilogy(r_bins_kpc, voldens_spher_data_Msun_pc3_data, 'k.', label = 'data')
		ax.semilogy(r_bins_kpc, voldens_spher_bestfit_Msun_pc3, color = self.spher_color, linestyle = '-', label = 'best fit')
		ax.set_ylabel('bulge volume density [M$_\odot$  pc$^{-3}$]')
		ax.set_xlabel('r [kpc]')
		ax.legend()
		fig.tight_layout()
		if safefigure == True:
			fig.savefig(self.plotdir + 'volume_dens_bulge_fit_data.png', dpi = 300, format = 'png' )
		plt.show()

	def plot_voldens_halo(self, safefigure = True, N = 25):
		voldens_halo_data_Msun_pc3_data, r_bins_kpc = self.voldens_halo_data(N)
		voldens_halo_bestfit_Msun_pc3 = self.voldens_halo_galpy(r_bins_kpc)
		fig,ax = plt.subplots(figsize = (8,8))
		ax.spines['top'].set_linewidth(1.5)
		ax.spines['bottom'].set_linewidth(1.5)
		ax.spines['left'].set_linewidth(1.5)
		ax.spines['right'].set_linewidth(1.5)
		ax.xaxis.set_tick_params(width=1.5)
		ax.yaxis.set_tick_params(width=1.5)
		ax.semilogy(r_bins_kpc, voldens_halo_data_Msun_pc3_data, 'k.', label = 'data')
		ax.semilogy(r_bins_kpc, voldens_halo_bestfit_Msun_pc3, color = self.halo_color, linestyle ='-', label = 'best fit')
		ax.set_ylabel('halo volume density [M$_\odot$  pc$^{-3}$]')
		ax.set_xlabel('r [kpc]')
		ax.legend()
		fig.tight_layout()
		if safefigure == True:
			fig.savefig(self.plotdir + 'volume_dens_halo_fit_data.png', dpi = 300, format = 'png' )
		plt.show()




