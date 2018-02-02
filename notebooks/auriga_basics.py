import numpy as np

from gadget import gadget_readsnap
from gadget_subfind import load_subfind

def eat_snap_and_fof(level, halo_number, snapnr, snappath, loadonlytype=[4],
                     haloid=0, galradfac=0.1, verbose=True, rotate_disk=True, 
                     use_principal_axis=True, euler_rotation=False, 
                     use_cold_gas_spin=False, do_rotation=True):
    """ Method to eat an Auriga snapshot, given a level/halo_number/snapnr.
        Subfind has been executed 'on-the-fly', during the simulation run.

        @param level: level of the Auriga simulation (3=high, 4='normal' or 5=low).
            Level 3/5 only for halo 6, 16 and 24. See Grand+ 2017 for details.
            Careful when level != 4 because directories may have different names.
        @param halo_number: which Auriga galaxy? See Grand+ 2017 for details.
            Should be an integer in range(1, 31)
        @param snapnr: which snapshot number? This is an integer, in most cases
            in range(1, 128) depending on the number of timesteps of the run.
            The last snapshot would then be 127. Snapshots are written at a
            certain time, but careful because the variable called time is actually
            the cosmological expansion factor a = 1/(1+z). For example, snapnr=127
            has s.time = 1, which corresponds to a redshift of ~0. This makes sense
            because this is the last snapshot and the last snapshot is written at
            redshift zero
        @param snappath: full path to the level/halo directory that contains
            all of the simulation snapshots
        @param loadonlytype: which particle types should be loaded? This should
            be a list of integers. If I'm not mistaken, the options are:
            0 (gas), 1 (halo), 2 (disk), 3 (bulge), 4 (stars), 5 (black holes).
            So to get the dark matter: load particles 1/2/3. In zoom-simulations
            particletype 3 may be used for low-resolution particles in the outer
            regions and they should not be present in (and contaminating) the
            inner region. I'm not too sure of the latter though.
        @param haloid: the ID of the SubFind halo. In case you are interested
            in the main galaxy in the simulation run: set haloid to zero.
            This was a bit confusing to me at first because a zoom-simulation run
            of one Auriga galaxy is also referred to as 'halo', see halo_number.
        @param galradfac: the radius of the galaxy is often used to make cuts in
            the (star) particles. It seems that in general galrad is set to 10%
            of the virial radius R200 of the DM halo that the galaxy sits in. The
            disk does seem to 'end' at 0.1R200.
        @param verbose: boolean to print some information

        @return: two-tuple (s, sf) where s is an instance of the gadget_snapshot
            class, and sf is an instance of the subfind class. See Arepo-snap-util,
            gadget_snap.py respectively gadget_subfind.py """

    # Eat the subfind friend of friends output
    sf = load_subfind(snapnr, dir=snappath)

    # Eat the Gadget snapshot
    s = gadget_readsnap(snapnr, snappath=snappath, lazy_load=True,
        subfind=sf, loadonlytype=loadonlytype)
    s.subfind = sf

    # Sets s.(sub)halo. This allows selecting the halo, e.g. 0 (main 'Galaxy')
    s.calc_sf_indizes(s.subfind)
    # Note that selecting the halo now rotates the disk using the principal axis.
    # rotate_disk is a general switch which has to be set to True to rotate.
    # To then actually do the rotation, do_rotation has to be True as well.
    # Within rotate_disk there are three methods to handle the rotation. Choose
    # one of them, but see the select_halo method for details.
    s.select_halo( s.subfind, haloid=haloid, galradfac=galradfac,
        rotate_disk=rotate_disk, use_principal_axis=use_principal_axis, 
        euler_rotation=euler_rotation, use_cold_gas_spin=use_cold_gas_spin, 
        do_rotation=do_rotation)

    # Sneak some more info into the s instance
    s.halo_number = halo_number
    s.level = level
    s.snapnr = snapnr
    s.haloid = haloid

    # This means that galrad is 10 % of R200 (200*rho_crit definition)
    s.galrad = galradfac * sf.data['frc2'][haloid]  # frc2 = Group_R_Crit200

    if verbose:
        print("\ngalrad  : {0}".format(s.galrad))
        print("redshift: {0}".format(s.redshift))
        print("time    : {0}".format(s.time))
        print("center  : {0}\n".format(s.center))

    return s, sf

def get_cartesian_vectors(s, sf, mask, kpc = True):
    x,  y,  z  = s.pos[::,2][mask], s.pos[::,1][mask], s.pos[::,0][mask]
    vx, vy, vz = s.vel[::,2][mask], s.vel[::,1][mask], s.vel[::,0][mask]

    rxyz = s.r()[mask]
    rxy = np.sqrt(x**2 + y**2)
    if kpc == True:
        (x, y, z), (vx, vy, vz), rxyz, rxy = (1000.*x, 1000.*y, 1000.*z), (1000.*vx, 1000.*vy, 1000.*vz), 1000.*rxyz, 1000.*rxy 
    return (x, y, z), (vx, vy, vz), rxyz, rxy

def get_cylindrical_vectors(s, sf, mask, kpc = True):
    (x, y, z), (vx, vy, vz), rxyz, rxy = get_cartesian_vectors(s, sf, mask, kpc)

    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    z = z
    
    vR = np.sqrt(vx**2 + vy**2)
    vphi = np.arctan2(vy, vx)
    vz = vz
    
    return (R, phi, z), (vR, vphi, vz)