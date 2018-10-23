"""
NAME: decomposition

PURPOSE: Class to decompose bulge and disk and to compare them to best fit galpy potentials.

HISTORY:  22-10-2018 - Written - Milanov (ESO)
"""

from areposnap.gadget import gadget_readsnap
from areposnap.gadget_subfind import load_subfind

from auriga_basics import *

from galpy.potential import MiyamotoNagaiPotential, NFWPotential, HernquistPotential

import numpy as np

from astropy import units as u

from matplotlib import pyplot as plt



class decomposition():
	def __init__(self, machine = 'magny'):
		self.machine = machine

		if self.machine == 'magny':
			self.filedir = "/home/extmilan/masterthesis/files/"
			self.basedir = "/hits/universe/GigaGalaxy/level4_MHD/"
			self.plotdir = "/home/extmilan/masterthesis/plots/"
		elif self.machine == 'mac':
			self.filedir = "/Users/smilanov/Documents/masterthesis/auriga_files/files/"
			self.basedir = "/Users/smilanov/Desktop/Auriga/level4/"
			self.plotdir = "/Users/smilanov/Documents/masterthesis/auriga_files/plots/"
		else:
			raise NotADirectoryError
            
		print('Load snapshot.')
		self._load_snapshot()
		print("Carry out decomposition.")
		self._decomp()
		print('Calculate disk indices.')
		self._get_disk_indices()
		print('Calculate spheroid indices.')
		self._get_spheroid_indices()
		print('Load positions and masses of simulation data.')
		self._get_positions()
		self._get_masses()
		print('Import galpy parameters.')
		self._get_galpy_parameters()
		print("Setup galpy potential.")
		self._setup_galpy_potential()


	def _get_galpy_parameters(self, inputfile = None):
		if inputfile == None:
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


	def _load_snapshot(self, halo_number = 24, startsnap = 127, endsnap = 128):

			
		#### path = /hits/universe/GigaGalaxy/level4_MHD/halo_24/output/*
		level = 4

		for halo_number in [halo_number]:  # range(1, 31):
			halodir = self.basedir+"halo_{0}/".format(halo_number)
			snappath = halodir+"output/"

			for snapnr in range(startsnap, endsnap, 1):
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

#_____function that sets-up galpy potential_____
	def _setup_galpy_potential(self):
		#test input:
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
        
		nfw_mass = np.sum(self.s.mass[((self.s.type == 1) + (self.s.type == 2) + (self.s.type == 3)) * (self.s.r()<=self.s.galrad)])* 1e10 * u.Msun
		hb_mass = 10**10*np.sum(self.s.mass[self.i_spher][self.s.r()[self.i_spher] <= self.s.galrad])*u.Msun
		#setup potential:
		self.disk = MiyamotoNagaiPotential(amp=10**10*np.sum(self.s.mass[self.i_disk][self.i_r_in])*u.Msun,
                                           a=self.a_MND_kpc*u.kpc, b=self.b_MND_kpc*u.kpc,)
		self.halo = NFWPotential(amp = nfw_mass, a = self.a_NFWH_kpc*u.kpc)
		self.bulge = HernquistPotential(amp=hb_mass,
                                        a = self.a_HB_kpc) 
		 
		self.pot = [self.disk,self.halo,self.bulge]
		#return [disk,halo,bulge], disk, halo, bulge


	def _decomp(self, circ_val = 0.7, plotter = False, include_zmax = False, zmax = 0.0005, Gcosmo = 43.0071):
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

		else:
			idisk, = np.where((eps2 >= circ_val)) 
			disk_ID = ID[istars][iensort][idisk]
			
		ispheroid = np.where(eps2 < circ_val)
		spheroid_ID = ID[istars][iensort][ispheroid]
		
		if plotter == True:
			fig, ax = plt.subplots(1, 1, figsize=(8,6))
			
			ydatatot, edges = np.histogram( eps2, bins=100, weights=smass, range=[-1.7,1.7] )
			xdatatot = 0.5 * (edges[1:] + edges[:-1])
			xdata = xdatatot

			ydatad, edgesd = np.histogram( eps2[idisk], bins=100, weights=smass[idisk], range=[-1.7,1.7] )
			ydatas, edgess = np.histogram( eps2[ispheroid], bins=100, weights=smass[ispheroid], range=[-1.7,1.7] )

			ax.fill( xdata, ydatad, fc='b', alpha=0.5, fill=True, lw=0, label='disc' )
			ax.fill( xdata, ydatas, fc='y', alpha=0.5, fill=True, lw=0, label='spheroid' )
			
			ax.plot( xdatatot, ydatatot, 'k', label='total' )
			ax.legend()
			ax.set_xlabel('$\epsilon$')
			plt.show()
			
		self.disk_IDs	  = disk_ID
		self.spheroid_IDs = spheroid_ID


	def _get_disk_indices(self):
		self.i_disk = np.isin(self.s.id, self.disk_IDs)

	def _get_spheroid_indices(self):
		self.i_spher = np.isin(self.s.id, self.spheroid_IDs)

	def _get_positions(self):
		self.r_kpc = 1000. * self.s.r()[self.i_disk]
		self.x_kpc = 1000. * self.s.pos[:, 2][self.i_disk]
		self.y_kpc = 1000. * self.s.pos[:, 1][self.i_disk]
		self.z_kpc = 1000. * self.s.pos[:, 0][self.i_disk]
		self.R_kpc = np.sqrt(self.x_kpc**2 + self.y_kpc**2)
		# keep it within galrad
		self.i_r_in = self.r_kpc <= (1000. * self.s.galrad)
	 
		# keep it within zmin - zmax range
		self.i_z_in = (self.z_kpc >= -5.) * (self.z_kpc <= 5.)

	def _get_masses(self):
		self.masses_10msun = self.s.mass[self.i_disk][self.i_r_in * self.i_z_in]

	def surfdens_data(self, N = 25):
		mass_hist, R_bin_edges = np.histogram(self.R_kpc[self.i_r_in * self.i_z_in], weights = self.masses_10msun, bins = N)
		area = np.pi * (R_bin_edges[1:]**2 - R_bin_edges[:-1]**2)
		rho_Msun_pc2 = 1e4 * mass_hist / area
		R_mean_kpc = R_bin_edges[:-1] + 1./2. * (R_bin_edges[1:] - R_bin_edges[:-1])

		return(rho_Msun_pc2, R_mean_kpc)

	def surfdens_galpy(self, R_bins_kpc, z_extend_kpc = 5.):

		surfdens_bestfit = np.zeros(len(R_bins_kpc))
		for i, item in enumerate(R_bins_kpc):
			surfdens_bestfit[i] = self.disk.surfdens(item * u.kpc, z_extend_kpc * u.kpc) 

		return(surfdens_bestfit)

	def circvel_galpy(self):
		pass

	def voldens_galpy(self):
		pass

	def circvel_data(self):
		pass
	def voldens_data(self):
		pass

	def plot_surfdens(self, N = 25):
		surfdens_data_Msun_pc2_data, R_bins_kpc = self.surfdens_data(N)
		surfdens_bestfit_Msun_pc2 = self.surfdens_galpy(R_bins_kpc)
		fig,ax = plt.subplots(figsize = (8,8))
		ax.plot(R_bins_kpc, surfdens_data_Msun_pc2_data, 'k.', label = 'data')
		ax.plot(R_bins_kpc, surfdens_bestfit_Msun_pc2, 'r-', label = 'best fit')
		ax.set_ylabel('surface density [$M_{\{odot}} / pc^2$]', fontsize = 22)
		ax.set_xlabel('R [kpc]', fontsize = 22)
		ax.legend()
		fig.tight_layout()
		fig.savefig(self.plotdir + 'surface_dens_disk_fit_data.png', dpi = 300, format = 'png' )
		plt.show()

	def plot_cirvel(self):
		pass
	def plot_voldens(self):
		pass
		




