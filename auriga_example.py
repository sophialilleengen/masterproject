#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example script to read in an Auriga simulation snapshot """

import numpy
import matplotlib
from matplotlib import pyplot
# pyplot.switch_backend('agg')

from gadget import gadget_readsnap
from gadget_subfind import load_subfind

__author__ = "Timo Halbesma"
__email__ = "halbesma@MPA-Garching.MPG.DE"


ZSUN = 0.0127

ELEMENTS = { 'H':0, 'He':1, 'C':2, 'N':3, 'O':4,
            'Ne':5, 'Mg':6, 'Si':7, 'Fe':8 }

# from Asplund et al. (2009) Table 5
SUNABUNDANCES = { 'H':12.0, 'He':10.98, 'C':8.47, 'N':7.87, 'O':8.73,
                  'Ne':7.97, 'Mg':7.64, 'Si':7.55, 'Fe':7.54 }


def p2(a):
    return ((a)*(a))


def example_plot(s, sf):
    """ Generate arXiv:1708.03635 Fig. 1 (right), but for Auriga. The
        authors use observations of stars in RAVE-TGAS and assign stars
        to the 'halo' if |z| > 1.5 kpc, and to the disk if |z| < 1.5 kpc.
        Here z is the direction along the height of the disk. This plot
        shows a histogram of the [Fe/H] for the disk+halo. """

    # Make a cut of the stars
    istars, = numpy.where(
        # Example to cut out an anulus of |r-rsun| < 2 kpc where rsun=8kpc
        # (s.r() < 10./1000) & (s.r() > 6./1000) & (s.age > 0.)

        # Criteria to select stars within the galactic radius that are
        # associated with the main galaxy
        (s.r() < s.galrad) & (s.r() > 0.) & (s.age > 0.)
        & (s.type == 4) & (s.halo == s.haloid)
    )

    # Criteria to select |x| > 1.5 kpc. The x-direction seems to be
    # the direction along the height of the disk. Note that I use
    # 1.5/1000. This is because the code internal unit length is Mpc.
    outside_disk, = numpy.where(numpy.abs(s.pos[::,0]) > 1.5/1000)
    inside_disk, = numpy.where(numpy.abs(s.pos[::,0]) < 1.5/1000)

    halo = numpy.intersect1d(istars, outside_disk)
    disk = numpy.intersect1d(istars, inside_disk)

    # Clean negative and zero values of gmet to avoid RuntimeErrors later on
    # (e.g. dividing by zero)
    s.data['gmet'] = numpy.maximum( s.data['gmet'], 1e-40 )

    # Here we compute [Fe/H] for the 'halo'. See arXiv:1708.03635 why
    # we adopt this criterion for disk and halo. Anyway, the subgrid model
    # tracks 9 species, see the global variable ELEMENTS. Here we select
    # Fe and H from the data, then scale to Solar.
    metal_halo = numpy.zeros( [numpy.size(halo),2] )
    metal_halo[:, 0] = s.data['gmet'][halo][:, ELEMENTS['Fe']]
    metal_halo[:, 1] = s.data['gmet'][halo][:, ELEMENTS['H']]
    feabund_halo = numpy.log10( metal_halo[:,0] / metal_halo[:,1] / 56. ) - \
        (SUNABUNDANCES['Fe'] - SUNABUNDANCES['H'])
    # Mask to use the bins in the histogram in the range -3, 2.
    mask_halo, = numpy.where((feabund_halo > -3) & (feabund_halo < 2))

    # Repeat, but for a different cut in star particles (the 'disk').
    metal_disk = numpy.zeros( [numpy.size(disk),2] )
    metal_disk[:, 0] = s.data['gmet'][disk][:, ELEMENTS['Fe']]
    metal_disk[:, 1] = s.data['gmet'][disk][:, ELEMENTS['H']]
    feabund_disk = numpy.log10( metal_disk[:,0] / metal_disk[:,1] / 56. ) - \
        (SUNABUNDANCES['Fe'] - SUNABUNDANCES['H'])
    mask_disk, = numpy.where((feabund_disk > -3) & (feabund_disk < 2))

    # Plot in the same style as the paper.
    fig, ax = pyplot.subplots()

    pyplot.hist(feabund_disk[mask_disk], bins=64, alpha=0.5, normed=True,
        color="blue", label="|x| < 1.5 kpc")
    pyplot.hist(feabund_halo[mask_halo], bins=64, alpha=0.5, normed=True,
        color="red", label="|x| > 1.5 kpc")
    ax.set_xlim(-3.1, 0.95)
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel(r"$f($[Fe/H]$)$")

    ax.text(0.01, 0.94, "Au{0}-{1} Star [Fe/H] Distribution"\
            .format(s.level, s.halo_number), weight="bold",
            fontsize=16, transform=pyplot.gca().transAxes)
    pyplot.legend(loc="center left", frameon=False)
    pyplot.show()
    pyplot.savefig("auriga_"+
        "{0}_halo{1}_snapnr{2}_disk-vs-halo_metal_histogram.pdf"\
        .format(s.level, s.halo_number, s.snapnr))
    pyplot.close()


def eat_snap_and_fof(level, halo_number, snapnr, snappath, loadonlytype=[4],
                     haloid=0, galradfac=0.1, verbose=True):
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
        rotate_disk=True, use_principal_axis=True, euler_rotation=False,
        use_cold_gas_spin=False, do_rotation=False )

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


if __name__ == "__main__":
    print("Reading in an Auriga simulation snapshot.\n")

    level = 4
    basedir = "/hits/universe/GigaGalaxy/level{0}_MHD/".format(level)
    for halo_number in [24]:  # range(1, 31):
        halodir = basedir+"halo_{0}/".format(halo_number)
        snappath = halodir+"output/"
        for snapnr in range(127, 128, 1):
            print("level   : {0}".format(level))
            print("halo    : {0}".format(halo_number))
            print("snapnr  : {0}".format(snapnr))
            print("basedir : {0}".format(basedir))
            print("halodir : {0}".format(halodir))
            print("snappath: {0}\n".format(snappath))

            s, sf = eat_snap_and_fof(level, halo_number, snapnr, snappath)
            example_plot(s, sf)
