        """
        NAME:
            
        PURPOSE:
            
        INPUT:
            
        OUTPUT:
            
        HISTORY:
            2018-0X-XX - Written (Milanov, ESO)
        """

class actions:
    def __init__(self, level, halo_number, snapnr, snappath, types = [4], haloid = 0, galradfac = 0.1, verbose = True):
        """
        NAME:
            __init__
        PURPOSE:
            initialize a potential object for a snapshot
        INPUT:
            level       = level of the Auriga simulation (3=high, 4='normal' or 5=low).
                          Level 3/5 only for halo 6, 16 and 24
            halo_number = Number of simulation
            snapnr      = Number of snap in [0:127]
            snappath    = path where to find the snapshot data
            types       = list of types ( 0: gas, 1-3: DM (halo/disk/bulge), 4: stars, 5: BHs)
                          default: stars 
            haloid      = the ID of the SubFind halo; default = 0
            galradfac   = the radius of the galaxy is often used to make cuts in
                          the (star) particles. It seems that in general galrad is set to 10%
                          of the virial radius R20
        OUTPUT:
            instance
        HISTORY:
            2018-08-08 - Written (Milanov, ESO)
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
      
        
    def setup_galpy_potential(self, a_MND_kpc, b_MND_kpc, a_NFWH_kpc, a_HB_kpc, n_MND, n_NFWH, n_HB):

        #test input:
        if (a_MND_kpc <= 0.) or (b_MND_kpc <= 0.) or (a_NFWH_kpc <= 0.) or (a_HB_kpc <= 0.) \
           or (n_MND <= 0.) or (n_NFWH <= 0.) or (n_HB <= 0.) or (n_MND >= 1.) or (n_NFWH >= 1.) or (n_HB >= 1.):
            raise ValueError('Error in setup_galpy_potential: '+\
                             'The input parameters for the scaling profiles do not correspond to a physical potential.')
        if np.fabs(n_MND + n_NFWH + n_HB - 1.) > 1e-7:
            raise ValueError('Error in setup_galpy_potential: '+\
                             'The sum of the normalization does not add up to 1.')

        #trafo to galpy units:
        a_MND  = a_MND_kpc / _REFR0_kpc
        b_MND  = b_MND_kpc / _REFR0_kpc
        a_NFWH = a_NFWH_kpc / _REFR0_kpc
        a_HB   = a_HB_kpc / _REFR0_kpc

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
        
        return [disk, halo, bulge]

    def actions(self, pot_galpy, IDs, R0_kpc, v0_kms):
        # create a mask of all GCs which survive until the end
        gcmask = np.isin(s.id, IDs)

        # get position and velocities of all selected GCs & convert to galpy units
        (R_kpc, phi_rad, z_kpc), (vR_kms, vphi_kms, vz_kms) = get_cylindrical_vectors(self.s, self.sf, gcmask)
        # convert physical to galpy units by dividing by REF vals
        R_galpy, vR_galpy, vT_galpy, z_galpy, vz_galpy = R_kpc / R0_kpc, vR_kms / v0_kms, vphi_kms / v0_kms, z_kpc / R0_kpc, vz_kms / v0_kms

        # estimate Delta of the Staeckel potential
        delta = 0.45
        delta = estimateDeltaStaeckel(pot_galpy, R_galpy, z_galpy)
        # CHECK HOW BIG INFLUENCE OF DELTA IS

        # set up the actionAngleStaeckel object
        aAS = actionAngleStaeckel(
                pot   = pot_galpy,  # potential
                delta = delta,      # focal length of confocal coordinate system
                c     = True        # use C code (for speed)
                )

        jR_galpy, lz_galpy, jz_galpy = aAS(R_galpy, vR_galpy, vT_galpy, z_galpy, vz_galpy)
        jR_kpckms, lz_kpckms, jz_kpckms = jR_galpy * R0_kpc * v0_kms, lz_galpy * R0_kpc * v0_kms, jz_galpy * R0_kpc * v0_kms

        return(jR_kpckms, lz_kpckms, jz_kpckms)
    
    def actions_cornerplot(self, jR_kpckms, lz_kpckms, jz_kpckms):
        data = np.vstack([jR_kpckms, lz_kpckms, jz_kpckms])
        labels = ['jR [kpc km/s]', 'lz [kpc km/s]', 'jz [kpc km/s]']
        figure = corner.corner(data.transpose(), labels = labels, plot_contours = 1, color = color, range =  [(0.,12500.), (-14000.,14000.),(0., 5000.)])
