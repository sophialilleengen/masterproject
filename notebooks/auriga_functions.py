import numpy as np
import matplotlib.pyplot as plt

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