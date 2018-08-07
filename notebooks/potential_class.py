        """
        NAME:
            
        PURPOSE:
            
        INPUT:
            
        OUTPUT:
            
        HISTORY:
            2018-0X-XX - Written (Milanov, ESO)
        """

## import general modules
import numpy as np
from scipy import optimize as opt

## import galpy modules
from galpy.potential import evaluatePotentials, MiyamotoNagaiPotential, NFWPotential, HernquistPotential

## import auriga modules
from auriga_basics import *
from auriga_functions import *

class potential:
    def __init__(self, level, halo_number, snapnr, snappath, types = [1,2,3,4], haloid = 0, galradfac = 0.1, verbose = True):
        """
        NAME:
            __init__
        PURPOSE:
            initialize a potential object for a snapshot
        INPUT:
            level       = level of the Auriga simulation (3=high, 4='normal' or 5=low).
                          Level 3/5 only for halo 6, 16 and 24
            halo_number = Number of simulation
            snapnr      = Bumber of snap in [0:127]
            snappath    = path where to find the snapshot data
            types       = list of types ( 0: gas, 1-3: DM (halo/disk/bulge), 4: stars, 5: BHs)
                          default: DM + stars
            haloid      = the ID of the SubFind halo; default = 0
            galradfac   = the radius of the galaxy is often used to make cuts in
                          the (star) particles. It seems that in general galrad is set to 10%
                          of the virial radius R20
        OUTPUT:
            instance
        HISTORY:
            2018-07-23 - Written (Milanov, ESO)
        """
        
        try:
            self.s, self.sf = eat_snap_and_fof(level, halo_number, snapnr, snappath, loadonlytype= types, 
            haloid=haloid, galradfac=galradfac, verbose=verbose)    
        except KeyError:
            print('\n\n', snapnr, 'not read in.\n\n')
            continue
            
        try: 
            # Clean negative and zero values of gmet to avoid RuntimeErrors
            # later on (e.g. dividing by zero)
            self.s.data['gmet'] = np.maximum( self.s.data['gmet'], 1e-40 )
        except:
            continue
            
        return None
      
    def disk_dens_data(self, R_kpc, z_kpc, dR_kpc, dz_kpc, M_disk_stars_10Msun): 
        """
        NAME:
            disk_dens_data
        PURPOSE:
            calculate density of cylindrical binned stars in disk component of the simulation data
        INPUT:
            R_kpc: array of each star's radial distance to center in kpc
            z_kpc: array of each star's vertical distance to center in kpc
            dR_kpc: bin length in kpc
            dz_kpc: bin height in kpc
            M_disk_stars_10Msun: array of each star's mass in 10^10 M_sun
        OUTPUT:
            rho_10Msun_kpc3: binned density in stars in disk in 10^10 M_sun / kpc^3
            Rbins_kpc: radial edges of bins in kpc
            zbins_kpc: vertical edges of bins in kpc
        HISTORY:
            2018-0X-XX - Written (Milanov, ESO)
        """
        Rmin_kpc, Rmax_kpc = np.min(R_kpc), np.max(R_kpc)
        zmin_kpc, zmax_kpc = np.min(z_kpc), np.max(z_kpc)

        Rbins, zbins = np.arange(Rmin_kpc, Rmax_kpc, dR_kpc), np.arange(zmin_kpc, zmax_kpc, dz_kpc)
        mbins, volbins = np.zeros((len(zbins), len(Rbins))), np.zeros((len(zbins), len(Rbins))) 
        for i in range(len(zbins)):
            for j in range(len(Rbins)):
                inbin = (Rbins[j] <= R_kpc) & (R_kpc < (Rbins[j] + dR_kpc)) & (zbins[i] <= z_kpc) & (z_kpc < (zbins[i] + dz_kpc))
                mbins[i,j] = np.sum(M_disk_stars_10Msun[inbin])
                volbins[i,j] = np.pi * dz_kpc * (2. * Rbins[j] * dR_kpc + dR_kpc**2)

        rho_10Msun_kpc3 = mbins / volbins
        return(rho_10Msun_kpc3, Rbins_kpc, zbins_kpc)   

    def disk_dens_MN(self, x, a_kpc, b_kpc):
        """
        NAME:
            disk_dens_MN
        PURPOSE:
            calculate the density of the disk according to the Miyamoto Nagai profile
        INPUT:
            x: R_kpc, z_kpc, Mass_tot in [kpc, kpc, 10^10 M_sun] either as single values (R_i, z_i, M_i) or as arrays of the same length (N)
            a_kpc: scale length of the MN disk in kpc
            b_kpc: scale height of the MN disk in kpc
        OUTPUT:
            rho_10Msun_kpc_3: density in 10^10 M_sun / kpc^3; either a single value as rho_i(R_i,z_i) or as a 1dim array with length N (?)
        HISTORY:
            2018-07-23 - Written (Milanov, ESO)
        """
        R_kpc, z_kpc, mass_10Msun = x[0], x[1], x[2]
        first_fac_kpc2 = (mass_10Msun * b_kpc**2) / (4. * np.pi) # * 10^10M_sun
        bzsqrt_kpc = np.sqrt(z_kpc**2 + b_kpc**2)
        abz_sqrt_kpc = (a_kpc + bzsqrt_kpc)
        numerator_kpc3 = a_kpc * R_kpc**2 + (a_kpc + 3. * bzsqrt_kpc) * abz_sqrt_kpc**2
        denominator_kpc8 = (R_kpc**2 + abz_sqrt_kpc**2)**(5./2.) * bzsqrt_kpc**3
        rho_10Msun_kpc3 = first_fac_kpc2 * numerator_kpc3 / denominator_kpc8 # * 10^10 M_sun if not specific density
        return(rho_10Msun_kpc_3)
    
    def fit_disk_MNdens(self, R_kpc, z_kpc, dR_kpc, dz_kpc, M_tot_disk_stars_10Msun, M_disk_stars_10Msun):
        """
        NAME:
            
        PURPOSE:
            
        INPUT:
            
        OUTPUT:
            
        HISTORY:
            2018-0X-XX - Written (Milanov, ESO)
        """
        rho_10Msun_kpc3_data, R_kpc_data, z_kpc_data = self.disk_dens_data(R_kpc, z_kpc, dR_kpc, dz_kpc, M_disk_stars_10Msun)
        
        side_x = R_data
        side_y = z_data
        X1, X2 = np.meshgrid(side_x, side_y)
        size = X1.shape
        x1_1d = X1.reshape((1, np.prod(size)))
        x2_1d = X2.reshape((1, np.prod(size)))
        size2 = x1_1d.shape
        x3_1d = np.repeat(M_tot_disk_stars_10Msun, size2[1])
        x3_1d = x3_1d.reshape((size2))
        xdata = np.vstack((x1_1d, x2_1d, x3_1d))
        z = rho_data
        Z = z.reshape(size2[1])
        ydata = Z
        popt, pcov = opt.curve_fit(MNdens, xdata, ydata)
        z_fit = self.disk_dens_MN(xdata, *popt)
        Z_fit = z_fit.reshape(size)
        
        '''
        plt.subplot(1, 2, 1)
        plt.title("Real Function")
        plt.pcolormesh(X1, X2, z, norm=LogNorm(), cmap=plt.get_cmap('plasma'))
        plt.axis('equal')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title("Fitted Function")
        plt.pcolormesh(X1, X2, Z_fit, norm=LogNorm(), cmap=plt.get_cmap('plasma'))
        plt.axis('equal')
        plt.colorbar()
        plt.tight_layout()

        plt.show()
        '''
        return(popt)
    
    def v_circ_disk_MN(self, R_kpc, z_kpc, a_MND_kpc, b_MND_kpc, M_disk_stars_tot_10Msun):
        """
        NAME:
            v_circ_disk_MN
        PURPOSE:
            calculate density of disk according to Miyamoto Nagai density
        INPUT:
            R_kpc: radial distance to center of galaxy in kpc (either single val or array)
            z_kpc: vertical distance to center of galaxy in kpc (either single val or array)
            a_MND_kpc: scale length of MN profile in kpc
            b_MND_kpc: scale height of MN profile in kpc
            M_disk_stars_tot_10Msun: total mass of all disk stars in 10^10 Msun
        OUTPUT:
            v_circ_MNdisk_kms: circular velocity at (R,z) in km/s (either single val or array)
        HISTORY:
            2018-07-27 - Written (Milanov, ESO)
        """
        denom = (R_kpc**2 + (a_MND_kpc + np.sqrt(z_kpc**2 + b_MND_kpc**2))**2)**(3./2.)
        v_circ_MNdisk_kms = R_kpc * np.sqrt(43.01e3* M_disk_stars_tot_10Msun / denom)
        return(v_circ_MNdisk_kms)        
    
    def v_circ_disk_data(self, disccirc_min = 0.9, disccirc_max = 1.0):
        """
        NAME:
            v_circ_disk_data
        PURPOSE:
            calculate circular velocity of disk 
        INPUT:
            disccirc_min: minimum circularity to be considered as being on a nearly circular orbit
            disccirc_max: maximum circularity to be considered as being on a nearly circular orbit
        OUTPUT:
            v_circ_datadisk_kms: circular velocity of the disk at (R,z) +- 0.5kpc in km/s
        HISTORY:
            2018-07-27 - Written (Milanov, ESO)
        """
        circ_disk_ID, spheroid_ID, bulge_ID, halo_ID = decomp(self.s, disccirc_min = disccirc_min, disccirc_max = disccirc_max)
        i_circ_disk = np.isin(s.id, circ_disk_ID)
        (R_kpc, phi, z_kpc), (vR_kms, vphi_kms, vz_kms) = get_cylindrical_vectors(self.s, self.sf, i_circ_disk, kpc = True):
        v_circ_datadisk_kms = np.mean(vphi_kms)
        return(v_circ_datadisk_kms)        
    
    def MND_potential(self, ):
        """
        NAME:
            
        PURPOSE:
            
        INPUT:
            
        OUTPUT:
            
        HISTORY:
            2018-0X-XX - Written (Milanov, ESO)
        """
        return None
    
    def spheroid_dens_data(self, M_spheroid_stars_10Msun, r_kpc, dr_kpc, log = True, n_log = 97):
        """
        NAME:
            spheroid_dens_data
        PURPOSE:
            calculate density of binned stars in spheroid component of the simulation data
        INPUT:
            M_spheroid_stars_10Msun: array of masses of each star in spheroid in 10^10 M_sun
            r_kpc: spherical distance to galaxy center in kpc 
            dr_kpc: if spheroid is binned lineary it is the depth of the shells
            log: if True, spheroid is binned logarithmically, if False then linear
            n_log: if spheroid bins are log, this is the number of bins
        OUTPUT:
            rho_10Msun_kpc3: density array of binned spheroid in 10^10 Msun / kpc^3 
            r_binned_out_kpc: edges of distance array in kpc
        HISTORY:
            2018-07-24 - Written (Milanov, ESO)
        """

        rmin_kpc, rmax_kpc = np.min(r_kpc), np.max(r_kpc)

        rbins_kpc     = np.arange(rmin_kpc, rmax_kpc, dr_kpc)
        rbins_log_kpc = np.logspace(np.log10(rmin_kpc), np.log10(rmax_kpc), n_log)
        mbins_10Msun  = np.zeros(len(rbins_kpc))
        volbins_kpc3  = np.zeros(len(rbins_kpc))
        if log == False:
            for i in range(len(rbins_kpc)):
                inbin = (rbins_kpc[i] <= r_kpc) & (r_kpc < (rbins_kpc[i] + dr_kpc)) # spherical shells
                mbins_10Msun[i] = np.sum(M_spheroid_stars_10Msun[inbin])
                volbins_kpc3[i] = 4./3. * np.pi * dr_kpc * (3. * rbins_kpc[i]**2 + 3. * rbins_kpc[i] * dr_kpc + dr_kpc**2)
            r_binned_out_kpc = rbins_kpc
        elif log == True:
            mbins_10Msun = np.zeros(len(rbins_log_kpc))
            volbins_kpc3 = np.zeros(len(rbins_log_kpc))
            dr_log_kpc = np.log(rbins_log_kpc[1]/rbins_log_kpc[0])
            for i in range(len(rbins_log_kpc)):
                inbin = (rbins_log_kpc[i] <= r_kpc) & (r_kpc < (rbins_log_kpc[i] + dr_log_kpc)) # spherical shells
                mbins_10Msun[i] = np.sum(M_spheroid_stars_10Msun[inbin])
                volbins_kpc3[i] = 4./3. * np.pi * dr_log_kpc * (3. * rbins_log_kpc[i]**2 + 3. * rbins_log_kpc[i] * dr_log_kpc + dr_log_kpc**2)
            r_binned_out_kpc = rbins_log_kpc
        
        rho_10Msun_kpc3 = mbins_10Msun / volbins_kpc3
        return(rho_10Msun_kpc3, r_binned_out_kpc)    
    
    def spheroid_dens_H(self, r_kpc, a_HB_kpc, amp_10Msun, log = True):
        """
        NAME:
            spheroid_dens_H
        PURPOSE:
            calculate the density at distance r according to the Hernquist potential
        INPUT:
            r_kpc: distance to center of galaxy in kpc - either single value or array
            a_HB_kpc: scale length of the spheroid in kpc
            amp_10Msun: amplitude of Hernquist potential in 10^10 Msun?
            log: if True, output is log(rho), if False it is rho
        OUTPUT:
            rho_out: density (either single value or array) as either log(rho) or rho 
        HISTORY:
            2018-07-24 - Written (Milanov, ESO)
        """
    
        rho_10Msun_kpc3 = amp_10Msun / (4. * np.pi * a_HB_kpc**3.) * (1. / ((r_kpc / a_HB_kpc) * (1. + r_kpc / a_HB_kpc)**3))
        if log == True:
            rho_out = np.log(rho_10Msun_kpc3)
        else: rho_out = rho_10Msun_kpc3 
        return(rho_out)

    def fit_spheroid_Hdens(self, M_spheroid_stars_10Msun, r_kpc, dr_kpc):
        """
        NAME:
            fit_spheroid_Hdens
        PURPOSE:
            fit a Hernquist profile to the density of the stellar spheroid
        INPUT:
            M_spheroid_stars_10Msun: array of masses of spheroid stars in 10^10 M_sun
            r_kpc: array of distances to center of these stars in kpc
            dr_kpc: bin size of spheroid in kpc if linear 
        OUTPUT:
            a_HB_kpc: scale length of Hernquist profile in kpc
        HISTORY:
            2018-07-24 - Written (Milanov, ESO)
        """
        rho_data_10Msun_kpc3, rbins_kpc = self.spheroid_dens_data(M_stars_10Msun, r_kpc, dr_kpc)
        rho_data_10Msun_kpc3 = np.log(rho_data_10Msun_kpc3)
        '''
        plt.plot(rbins_kpc, rho_data_10Msun_kpc3, 'r.')
        plt.xlabel('r[kpc]')
        plt.ylabel('rho [10^10Msun/kpc^3]')
        '''
        rho_data_10Msun_kpc3 = np.where(np.isinf(rho_data_10Msun_kpc3), rho_data_10Msun_kpc3[-2] ,rho_data_10Msun_kpc3)
        popt, pcov = opt.curve_fit(self.spheroid_dens_H, rbins_kpc, rho_data_10Msun_kpc3)
        a_HB_kpc = popt[0]
        '''
        plt.plot(rbins_kpc, Hernquistdens(rbins_kpc, popt[0], popt[1]))
        plt.show()
        '''
        return(a_HB_kpc)

    def v_circ_spheroid_H(r_kpc, a_HB_kpc, M_spheroid_stars_tot_10Msun):
        """
        NAME:
            v_circ_spheroid_H
        PURPOSE:
            calculate the circular velocity according to the Hernquist profile
        INPUT:
            r_kpc: distance to the center in kpc (either single val or array)
            a_HB_kpc: scale length of the Hernquist potential in kpc
            M_spheroid_stars_tot_10Msun: total mass of spheroid stars in 10^10 M_sun
        OUTPUT:
            v_circ_Hspheroid_kms: circular velocity at r in km/s (either single val or array)
        HISTORY:
            2018-07-27 - Written (Milanov, ESO)
        """
        v_circ_Hspheroid_kms = np.sqrt(43.0071e3* M_spheroid_stars_tot_10Msun * r_kpc) / (r_kpc + a_HB_kpc)
        return(v_circ_Hspheroid_kms)
        
    def v_circ_spheroid_data(r_kpc_spher, M_spheroid_stars_10Msun, n_shells = 25):
        """
        NAME:
            v_circ_spheroid_data
        PURPOSE:
            calculate the circular velocity in a spherical profile            
        INPUT:
            r_kpc_spher: distance of the spheroid stars to the center in kpc in array
            M_spheroid_stars_10Msun: mass of the spheroid stars in 10^10 M_sun in array
            n_shells: number of shells in which circular velocity is calculated
        OUTPUT:
            v_circ_dataspheroid_kms: circular velocity of the spheroid star data in km/s (array length = n_shells)
        HISTORY:
            2018-07-27 - Written (Milanov, ESO)
        """
        r_kpc = np.linspace(np.min(r_kpc_spher), np.max(r_kpc_spher), n_shells)
        v_circ_dataspheroid_kms = np.zeros(len(r_kpc))
        for i, item in enumerate(r_kpc):
            mass_mask = np.where((r_kpc_spher<=item))
            mass_10Msun  = np.sum(M_spheroid_stars_10Msun[mass_mask])
            v_circ_dataspheroid_kms[i] = np.sqrt(43.0071e3 * mass_10Msun / item)
        return(v_circ_dataspheroid_kms)

    def HB_potential(self, ):
        """
        NAME:
            
        PURPOSE:
            
        INPUT:
            
        OUTPUT:
            
        HISTORY:
            2018-0X-XX - Written (Milanov, ESO)
        """
        return None
    
    
    def halo_dens_data(self, M_halo_DM_10Msun, r_kpc, dr_kpc, log = True, n_log = 97):
        """
        NAME:
            halo_dens_data
        PURPOSE:
            calculate density of binned DM particles in "halo" component of the simulation data
        INPUT:
            M_halo_DM_10Msun: array of masses of each DM particle in 10^10 M_sun
            r_kpc: spherical distance to galaxy center in kpc 
            dr_kpc: if halo is binned lineary it is the depth of the shells
            log: if True, halo is binned logarithmically, if False then linear
            n_log: if halo bins are log, this is the number of bins
        OUTPUT:
            rho_10Msun_kpc3: density array of binned halo in 10^10 Msun / kpc^3 
            r_binned_out_kpc: edges of distance array in kpc
        HISTORY:
            2018-07-24 - Written (Milanov, ESO)
        """

        rmin_kpc, rmax_kpc = np.min(r_kpc), np.max(r_kpc)

        rbins_kpc     = np.arange(rmin_kpc, rmax_kpc, dr_kpc)
        rbins_log_kpc = np.logspace(np.log10(rmin_kpc), np.log10(rmax_kpc), n_log)
        mbins_10Msun  = np.zeros(len(rbins_kpc))
        volbins_kpc3  = np.zeros(len(rbins_kpc))
        if log == False:
            for i in range(len(rbins_kpc)):
                inbin = (rbins_kpc[i] <= r_kpc) & (r_kpc < (rbins_kpc[i] + dr_kpc)) # spherical shells
                mbins_10Msun[i] = np.sum(M_halo_DM_10Msun[inbin])
                volbins_kpc3[i] = 4./3. * np.pi * dr_kpc * (3. * rbins_kpc[i]**2 + 3. * rbins_kpc[i] * dr_kpc + dr_kpc**2)
            r_binned_out_kpc = rbins_kpc
        elif log == True:
            mbins_10Msun = np.zeros(len(rbins_log_kpc))
            volbins_kpc3 = np.zeros(len(rbins_log_kpc))
            dr_log_kpc = np.log(rbins_log_kpc[1]/rbins_log_kpc[0])
            for i in range(len(rbins_log_kpc)):
                inbin = (rbins_log_kpc[i] <= r_kpc) & (r_kpc < (rbins_log_kpc[i] + dr_log_kpc)) # spherical shells
                mbins_10Msun[i] = np.sum(M_halo_DM_10Msun[inbin])
                volbins_kpc3[i] = 4./3. * np.pi * dr_log_kpc * (3. * rbins_log_kpc[i]**2 + 3. * rbins_log_kpc[i] * dr_log_kpc + dr_log_kpc**2)
            r_binned_out_kpc = rbins_log_kpc
        
        rho_10Msun_kpc3 = mbins_10Msun / volbins_kpc3
        return(rho_10Msun_kpc3, r_binned_out_kpc)    
    
    def halo_dens_NFW(self, r_kpc, a_NFWH_kpc, amp_10Msun, log = True):
        """
        NAME:
            halo_dens_NFW
        PURPOSE:
            calculate the density at distance r according to the NFW potential
        INPUT:
            r_kpc: distance to center of galaxy in kpc - either single value or array
            a_NFWH_kpc: scale length of the halo in kpc
            amp_10Msun: amplitude of NFW potential in 10^10 Msun?
            log: if True, output is log(rho), if False it is rho
        OUTPUT:
            rho_out: density (either single value or array) as either log(rho) or rho 
        HISTORY:
            2018-07-24 - Written (Milanov, ESO)
        """
    
        rho_10Msun_kpc3 = amp_10Msun / (4. * np.pi * a_NFWH_kpc**3.) * (1. / ((r_kpc / a_NFWH_kpc) * (1. + r_kpc / a_NFWH_kpc)**3))
        if log == True:
            rho_out = np.log(rho_10Msun_kpc3)
        else: rho_out = rho_10Msun_kpc3 
        return(rho_out)

    def fit_halo_NFWdens(self, M_halo_DM_10Msun, r_kpc, dr_kpc):
        """
        NAME:
            fit_halo_NFWdens
        PURPOSE:
            fit a NFW profile to the density of the DM halo
        INPUT:
            M_halo_DM_10Msun: array of masses of  halo DM particles in 10^10 M_sun
            r_kpc: array of distances to center of these DM particles in kpc
            dr_kpc: bin size of halo in kpc if linear 
        OUTPUT:
            a_NFWH_kpc: scale length of NFW profile in kpc
        HISTORY:
            2018-07-24 - Written (Milanov, ESO)
        """
        rho_data_10Msun_kpc3, rbins_kpc = self.halo_dens_data(M_halo_DM_10Msun, r_kpc, dr_kpc)
        rho_data_10Msun_kpc3 = np.log(rho_data_10Msun_kpc3)
        '''
        plt.plot(rbins_kpc, rho_data_10Msun_kpc3, 'r.')
        plt.xlabel('r[kpc]')
        plt.ylabel('rho [10^10Msun/kpc^3]')
        '''
        rho_data_10Msun_kpc3 = np.where(np.isinf(rho_data_10Msun_kpc3), rho_data_10Msun_kpc3[-2] ,rho_data_10Msun_kpc3)
        popt, pcov = opt.curve_fit(self.halo_dens_NFW, rbins_kpc, rho_data_10Msun_kpc3)
        a_NFWH_kpc = popt[0]
        '''
        plt.plot(rbins_kpc, Hernquistdens(rbins_kpc, popt[0], popt[1]))
        plt.show()
        '''
        return(a_NFWH_kpc)

    def v_circ_halo_data(r_kpc_spher, M_spheroid_stars_10Msun, n_shells = 25):
        """
        NAME:
            v_circ_spheroid_data
        PURPOSE:
            calculate the circular velocity in a spherical profile            
        INPUT:
            r_kpc_spher: distance of the spheroid stars to the center in kpc in array
            M_spheroid_stars_10Msun: mass of the spheroid stars in 10^10 M_sun in array
            n_shells: number of shells in which circular velocity is calculated
        OUTPUT:
            v_circ_dataspheroid_kms: circular velocity of the spheroid star data in km/s (array length = n_shells)
        HISTORY:
            2018-07-27 - Written (Milanov, ESO)
        """
        r_kpc = np.linspace(np.min(r_kpc_spher), np.max(r_kpc_spher), n_shells)
        v_circ_dataspheroid_kms = np.zeros(len(r_kpc))
        for i, item in enumerate(r_kpc):
            mass_mask = np.where((r_kpc_spher<=item))
            mass_10Msun  = np.sum(M_spheroid_stars_10Msun[mass_mask])
            v_circ_dataspheroid_kms[i] = np.sqrt(43.0071e3 * mass_10Msun / item)
        return(v_circ_dataspheroid_kms)
    
    def NFW_potential(self, ):
        """
        NAME:
            
        PURPOSE:
            
        INPUT:
            
        OUTPUT:
            
        HISTORY:
            2018-0X-XX - Written (Milanov, ESO)
        """
        return None
    
    def v_circ_tot_func(r_kpc):
        """
        NAME:
            
        PURPOSE:
            
        INPUT:
            
        OUTPUT:
            
        HISTORY:
            2018-0X-XX - Written (Milanov, ESO)
        """
        try:
            test = len(r_kpc)
            types = ( s.type == 0) + (s.type == 1) + (s.type == 2) + (s.type == 3) + (s.type == 4)

            #mass_mask    = np.zeros(len(r_kpc))
            #mass_10Msun  = np.zeros(len(r_kpc))
            v_circ_shell = np.zeros(len(r_kpc))

            for i, item in enumerate(r_kpc):
                mass_mask    = np.where( (types) & (s.halo == 0) & ((1000. * s.r()) < (item)))
                mass_10Msun  = np.sum(s.mass[mass_mask])
                v_circ_shell[i] = np.sqrt(43.01e3 * mass_10Msun / item)

            v_circ =  v_circ_shell
        except:
            mass_10Msun = np.sum(s.mass[((1000. * s.r()) <= r_kpc) ] )
            v_circ = np.sqrt(43.01e3 * mass_10Msun / r_kpc)

        return(v_circ)

    def setup_galpy_potential(self, a_MND_kpc, b_MND_kpc, a_NFWH_kpc, a_HB_kpc, n_MND, n_NFWH, n_HB, _REFR0_kpc):
        """
        NAME:
            
        PURPOSE:
            
        INPUT:
            
        OUTPUT:
            
        HISTORY:
            2018-0X-XX - Written (Milanov, ESO)
        """
        #test input:
        if (a_MND_kpc <= 0.) or (b_MND_kpc <= 0.) or (a_NFWH_kpc <= 0.) or (a_HB_kpc <= 0.) \
           or (n_MND <= 0.) or (n_NFWH <= 0.) or (n_HB <= 0.) or (n_MND >= 1.) or (n_NFWH >= 1.) or (n_HB >= 1.):
            raise ValueError('Error in setup_galpy_potential: '+\
                             'The input parameters for the scaling profiles do not correspond to a physical potential.')
        if np.fabs(n_MND + n_NFWH + n_HB - 1.) > 1e-7:
            raise ValueError('Error in setup_galpy_potential: '+\
                             'The sum of the normalization does not add up to 1.')

        #trafo to galpy units:
        a_MND  = a_MND_kpc  / _REFR0_kpc
        b_MND  = b_MND_kpc  / _REFR0_kpc
        a_NFWH = a_NFWH_kpc / _REFR0_kpc
        a_HB   = a_HB_kpc   / _REFR0_kpc

        #setup potential:
        disk = MiyamotoNagaiPotential(
                    a = a_MND,
                    b = b_MND,
                    normalize = n_MND)
        halo = NFWPotential(
                    a = a_NFWH,
                    normalize = n_NFWH)
        bulge = HernquistPotential(
                    a = a_HB,
                    normalize = n_HB) 

        return [disk,halo,bulge]
    
    
    def rel_pot_error_scipydifferentialevolution(self, x,*args):
        """
        NAME:
            
        PURPOSE:
            
        INPUT:
            
        OUTPUT:
            
        HISTORY:
            2018-0X-XX - Written (Milanov, ESO)
        """
    
        # read fitting parameters:
        v0_tot_kms = x[0]
        a_NFWH_kpc = x[1]


        # read data:

        a_MND_kpc     = args[0]
        b_MND_kpc     = args[1]
        a_HB_kpc      = args[2]    
        v0_MND_kms    = args[3]
        v0_HB_kms     = args[4]
        R_kpc_data    = args[5]
        z_kpc_data    = args[6]
        pot_kms2_data = args[7]
        _REFR0_kpc    = args[8]


        v0_NFWH_kms = np.sqrt(v0_tot_kms**2 - v0_HB_kms**2 - v0_MND_kms**2)

        n_NFWH = v0_NFWH_kms**2 / v0_tot_kms**2
        n_HB   = v0_HB_kms**2   / v0_tot_kms**2
        n_MND  = v0_MND_kms**2  / v0_tot_kms**2
        # setup potential (and check if parameters are physical):
        try:
            pot_galpy_model = setup_galpy_potential(a_MND_kpc, b_MND_kpc, a_NFWH_kpc, a_HB_kpc, n_MND, n_NFWH, n_HB, _REFR0_kpc)
        except Exception as e:
            return np.inf

        # calculate potential values at (R,z) for this potential:
        #print(R_kpc_data)
        #print(z_kpc_data)
        #print(_REFR0_kpc)
        #print(v0_tot_kms)
        pot_kms2_model = evaluatePotentials(pot_galpy_model,
                                       R_kpc_data / _REFR0_kpc,
                                       z_kpc_data / _REFR0_kpc) * (v0_tot_kms)**2

        #calculate sum of relative error squares:
        err = np.sum(((pot_kms2_data - pot_kms2_model) / pot_kms2_model)**2)
        return err
    
    
    