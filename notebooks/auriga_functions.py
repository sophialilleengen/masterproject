import numpy as np
import matplotlib.pyplot as plt
import sys

from areposnap.gadget import gadget_readsnap
from areposnap.gadget_subfind import load_subfind

def dens(M = None, R_kpc = None, z_kpc = None, s = None, dR_kpc = None, dz_kpc = None, nbins = None):
    '''
    NAME: 
        dens
        
    PURPOSE:
        calculates the binned density in cylindrical coordinates
        
    INPUT:
        as input values either the combination of M, R_kpc and z_kpc is needed or just s
        M - mass array in 10^10 M_sun (same length as R_kpc and z_kpc) 
        R_kpc - radial distance array in kpc (same length as M and z_kpc)
        z_kpc - vertical height array in kpc (same length as M and R_kpc)
        s - Snapshot of simulation to be investigated
        dR_kpc - radial bin width in kpc
        dz_kpc - vertical bin width in kpc 
        
    OUTPUT:
        rho - density in 10^10 M_sun / kpc^3
        rho_arr_real, rho_arr_mean - needed for weights in histogram (might be wrong) in 10^10 M_sun / kpc^3
        Rbins, zbins - R and z bins (also for histogram)
        
    HISTORY:
        13-02-2018 - Written - Milanov (ESO)
        09-03-2018 - Modified: s as alternative input - Milanov (ESO)
    
    To do:
        - check how to weight the histogram properly
        - make tests that inputs are in instances I want
    '''
    if (M == None) * (R_kpc == None) * (z_kpc == None) * (s == None):
        sys.exit('Need either s or (M, R_kpc, z_kpc) as input.')
        
    elif (M == None) * (R_kpc == None) * (z_kpc == None) * (s != None):
        M = s.mass()
        R_kpc = 1000. * sqrt(s.pos[1]**2 + s.pos[2]**2) 
        z_kpc = 1000. * s.pos[0]    
        
    elif (s != None) * ((M != None) or (R_kpc != None) or (z_kpc != None)):
        M = s.mass()
        R_kpc = 1000. * sqrt(s.pos[1]**2 + s.pos[2]**2) 
        z_kpc = 1000. * s.pos[0]        
        
    Rmin_kpc, Rmax_kpc = np.min(R_kpc), np.max(R_kpc)
    zmin_kpc, zmax_kpc = np.min(z_kpc), np.max(z_kpc)
    
    if nbins != None:
        dR_kpc = (Rmax_kpc - Rmin_kpc) / nbins
        dz_kpc = (zmax_kpc - zmin_kpc) / nbins

    Rbins, zbins = np.arange(Rmin_kpc, Rmax_kpc, dR_kpc), np.arange(zmin_kpc, zmax_kpc, dz_kpc)
    mbins, volbins = np.zeros((len(Rbins), len(zbins))), np.zeros((len(Rbins), len(zbins))) 
    rho_arr_real, rho_arr_mean = np.zeros(len(R_kpc)), np.zeros(len(R_kpc))
    for i in range(len(Rbins)):
        for j in range(len(zbins)):
            inbin = (Rbins[i] <= R_kpc) & (R_kpc < (Rbins[i] + dR_kpc)) & (zbins[j] <= z_kpc) & (z_kpc < (zbins[j] + dz_kpc))
            mbins[i,j] = np.sum(M[inbin])
            volbins[i,j] = np.pi * dz_kpc * (2. * Rbins[i] * dR_kpc + dR_kpc**2)
            rho_arr_real[inbin] = mbins[i,j] / volbins[i,j] 
            rho_arr_mean[inbin] = (mbins[i,j] / volbins[i,j]) / len(inbin) 
    m_enc = np.sum(mbins, axis = 1)
    rho = mbins / volbins
    return(rho, rho_arr_real, rho_arr_mean, Rbins, zbins, volbins, m_enc)

def fitting_dens(M = None, R_kpc = None, z_kpc = None, s = None, dR_kpc = None, dz_kpc = None, nbins = None):
    '''
    NAME: 
        dens
        
    PURPOSE:
        calculates the binned density in cylindrical coordinates
        
    INPUT:
        as input values either the combination of M, R_kpc and z_kpc is needed or just s
        M - mass array in 10^10 M_sun (same length as R_kpc and z_kpc) 
        R_kpc - radial distance array in kpc (same length as M and z_kpc)
        z_kpc - vertical height array in kpc (same length as M and R_kpc)
        s - Snapshot of simulation to be investigated
        dR_kpc - radial bin width in kpc
        dz_kpc - vertical bin width in kpc 
        
    OUTPUT:
        rho - density in 10^10 M_sun / kpc^3
        rho_arr_real, rho_arr_mean - needed for weights in histogram (might be wrong) in 10^10 M_sun / kpc^3
        Rbins, zbins - R and z bins (also for histogram)
        
    HISTORY:
        13-02-2018 - Written - Milanov (ESO)
        09-03-2018 - Modified: s as alternative input - Milanov (ESO)
    
    To do:
        - check how to weight the histogram properly
        - make tests that inputs are in instances I want
    '''
    if (np.any(M == None)) * (np.any(R_kpc == None)) * (np.any(z_kpc == None)) * (s == None):
        print('1')
        sys.exit('Need either s or (M, R_kpc, z_kpc) as input.')
        
    elif (np.any(M == None)) * (np.any(R_kpc == None)) * (np.any(z_kpc == None)) * (s != None):
        print('2')
        M = s.mass()
        R_kpc = 1000. * sqrt(s.pos[1]**2 + s.pos[2]**2) 
        z_kpc = 1000. * s.pos[0]    
        
    elif (s != None) * ((np.any(M != None)) or (np.any(R_kpc != None)) or (np.any(z_kpc != None))):
        print('3')
        M = s.mass()
        R_kpc = 1000. * sqrt(s.pos[1]**2 + s.pos[2]**2) 
        z_kpc = 1000. * s.pos[0]        
    else:
        print('Mass, radius and height given as input.')
        
    Rmin_kpc, Rmax_kpc = np.min(R_kpc), np.max(R_kpc)
    zmin_kpc, zmax_kpc = np.min(z_kpc), np.max(z_kpc)

    if nbins != None:
        dR_kpc = (Rmax_kpc - Rmin_kpc) / nbins
        dz_kpc = (zmax_kpc - zmin_kpc) / nbins
    
    #print(Rmin_kpc, Rmax_kpc, zmin_kpc, zmax_kpc)
    Rbins, zbins = np.arange(Rmin_kpc, Rmax_kpc, dR_kpc), np.arange(zmin_kpc, zmax_kpc, dz_kpc)
    mbins, volbins = np.zeros((len(Rbins), len(zbins))), np.zeros((len(Rbins), len(zbins))) 
    for i in range(len(Rbins)):
        for j in range(len(zbins)):
            inbin = (Rbins[i] <= R_kpc) & (R_kpc < (Rbins[i] + dR_kpc)) & (zbins[j] <= z_kpc) & (z_kpc < (zbins[j] + dz_kpc))
            mbins[i,j] = np.sum(M[inbin])
            volbins[i,j] = np.pi * dz_kpc * (2. * Rbins[i] * dR_kpc + dR_kpc**2)
   
    m_enc = np.sum(mbins, axis = 1)
    rho = mbins / volbins
    return(rho, Rbins, zbins, volbins, m_enc)

def decomp(s, plotter = False, disccirc = 0.7, galrad = 0.1, Gcosmo = 43.0071):
    ID = s.id
    # get number of particles 
    na = s.nparticlesall

    # get mass and age of stars 
    mass = s.data['mass'].astype('float64')
    st = na[:4].sum(); en = st+na[4]
    age = np.zeros( s.npartall )
    # only stars will be given an age, for other particles: age = 0
    age[st:en] = s.data['age']

    # create masks for all particles / stars within given radius
    iall, = np.where( (s.r() < galrad) & (s.r() > 0.) )
    istars, = np.where( (s.r() < galrad) & (s.r() > 0.) & (s.type == 4) & (age > 0.) )

    # calculate radius of all particles within galrad, sort them, \
    # get their mass and calculate their cumulative mass sorted by their distance
    nstars = len( istars )
    nall   = len( iall )
    rr = np.sqrt( (s.pos[iall,:]**2).sum(axis=1) )
    msort = rr.argsort()
    mass_all = mass[iall]
    msum = np.zeros( nall )
    msum[msort[:]] = np.cumsum(mass_all[msort[:]])

    # get position, velocity, mass, type, potential, radius and age of all particles within galrad
    pos  = s.pos[iall,:].astype( 'float64' )
    vel = s.vel[iall,:].astype( 'float64' )
    mass = s.data['mass'][iall].astype( 'float64' )
    ptype = s.data['type'][iall]
    pot = s.data['pot'][iall].astype('float64')
    radius = np.sqrt( (s.pos[iall,:]**2).sum(axis=1) )
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
    star_age[:] = s.cosmology_get_lookback_time_from_a( age[nn], is_flat=True )

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

    # divide star mass by mass of all stars to use as weight in histogram
    smass /= smass.sum()
    # calculate eps = jz/j_z_max 
    eps2[:] /= jcmax[:]
    
    
    idisk = np.where(eps2 >= 0.7)
    ibulge = np.where(eps2 < 0.7)
    disk_ID = ID[istars][iensort][idisk]
    bulge_ID = ID[istars][iensort][ibulge]
    
    if plotter == True:
        fig, ax = plt.subplots(1, 1, figsize=(8,6))
        #fig.set_axis_labels( xlabel="$\epsilon$", ylabel="$\\rm{f(\epsilon)}$" )
        ax.set_xlabel("$\epsilon$")
        ax.set_ylabel("$\\rm{f(\epsilon)}$")
        ydatatot, edges = np.histogram( eps2, bins=100, weights=smass, range=[-1.7,1.7] )
        xdatatot = 0.5 * (edges[1:] + edges[:-1])
        xdata = xdatatot

        ydatad, edgesd = np.histogram( eps2[idisk], bins=100, weights=smass[idisk], range=[-1.7,1.7] )
        ydatab, edgesb = np.histogram( eps2[ibulge], bins=100, weights=smass[ibulge], range=[-1.7,1.7] )

        ax.fill( xdata, ydatab, fc='r', alpha=0.5, fill=True, lw=0, label='spheroid' )
        ax.fill( xdata, ydatad, fc='b', alpha=0.5, fill=True, lw=0, label='disc' )
        ax.plot( xdatatot, ydatatot, 'k', label='total' )
        ax.legend()
    
    return(disk_ID, bulge_ID)