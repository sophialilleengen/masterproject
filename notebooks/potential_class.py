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


## import auriga modules
import auriga_basics import *

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
            
        PURPOSE:
            
        INPUT:
            
        OUTPUT:
            
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
        return(rho, Rbins_kpc, zbins_kpc)   

    def disk_dens_MN(self, x, a_kpc, b_kpc):
        """
        NAME:
            
        PURPOSE:
            
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
    
    def rho_data(M_stars_10Msun, r_kpc, dr_kpc, log = True):
        """
        NAME:
            
        PURPOSE:
            
        INPUT:
            
        OUTPUT:
            
        HISTORY:
            2018-0X-XX - Written (Milanov, ESO)
        """

        rmin_kpc, rmax_kpc = np.min(r_kpc), np.max(r_kpc)

        rbins_kpc = np.arange(rmin_kpc, rmax_kpc, dr_kpc)
        rbins_log_kpc = np.logspace(np.log10(rmin_kpc), np.log10(rmax_kpc), 97)
        mbins_10Msun = np.zeros(len(rbins_kpc))
        volbins_kpc3 = np.zeros(len(rbins_kpc))
        if log == False:
            for i in range(len(rbins_kpc)):
                inbin = (rbins_kpc[i] <= r_kpc) & (r_kpc < (rbins_kpc[i] + dr_kpc)) # spherical shells
                mbins_10Msun[i] = np.sum(M_stars_10Msun[inbin])
                volbins_kpc3[i] = 4./3. * np.pi * dr_kpc * (3. * rbins_kpc[i]**2 + 3. * rbins_kpc[i] * dr_kpc + dr_kpc**2)
        elif log == True:
            mbins_10Msun = np.zeros(len(rbins_log_kpc))
            volbins_kpc3 = np.zeros(len(rbins_log_kpc))
            dr_log_kpc = np.log(rbins_log_kpc[1]/rbins_log_kpc[0])
            for i in range(len(rbins_log_kpc)):
                inbin = (rbins_log_kpc[i] <= r_kpc) & (r_kpc < (rbins_log_kpc[i] + dr_log_kpc)) # spherical shells
                mbins_10Msun[i] = np.sum(M_stars_10Msun[inbin])
                volbins_kpc3[i] = 4./3. * np.pi * dr_log_kpc * (3. * rbins_log_kpc[i]**2 + 3. * rbins_log_kpc[i] * dr_log_kpc + dr_log_kpc**2)


        rho_10Msun_kpc3 = mbins_10Msun / volbins_kpc3
        return(rho_10Msun_kpc3, rbins_log_kpc)    
    
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
    
    
