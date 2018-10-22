"""
NAME: decomposition

PURPOSE: Class to decompose bulge and disk and to compare them to best fit galpy potentials.

HISTORY:  22-10-2018 - Written - Milanov (ESO)
"""

class decomposition():
	def __init__(self):

	def decomp(s, circ_val = 0.6, plotter = False, include_zmax = False, zmax = 0.0005):
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
	    Radius = np.sqrt((s.pos[iall,2]**2 + s.pos[iall,1]**2))
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
	    nn, = np.where((ptype[:] == 4) & (age[:] > 0.)) 

	    # divide star mass by mass of all stars to use as weight in histogram
	    smass /= smass.sum()
	    # calculate eps = jz/j_z_max 
	    eps2[:] /= jcmax[:]

	    Radius_Mpc = s.r()[istars][iensort]
	    
	    if include_zmax == True:
	        idisk, = np.where((eps2 >= circ_val) & (abs(s.pos[:,0][istars][iensort]) < zmax))
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
	        
	    plt.hist2d
	    print(0)
	    return(disk_ID, spheroid_ID)

	print('start')


	def get_disk_indices(disk_IDs):
	    print('Comparing indices.')
	    i_disk = np.isin(s.id, disk_IDs)
	    print('Indices calculated.')
	    return(i_disk)

	def get_spheroid_indices(spheroid_IDs):
	    print('Comparing indices.')
	    i_spher = np.isin(s.id, spheroid_IDs)
	    print('Indices calculated.')
	    return(i_spher)

	if start_notebook == True:
	    disk_IDs, spheroid_IDs = decomp(s, 0.7, plotter = False)#, include_zmax=True, zmax=0.005)
	    i_disk_s_all = get_disk_indices(disk_IDs)
	    start_notebook = False