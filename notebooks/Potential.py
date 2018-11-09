from areposnap.gadget import gadget_readsnap
from areposnap.gadget_subfind import load_subfind

from auriga_basics import *
from decomposition import *

from galpy.potential import MiyamotoNagaiPotential, NFWPotential, HernquistPotential
from galpy.potential.plotRotcurve import vcirc
from galpy.util import bovy_conversion
import numpy as np

from scipy import stats

from astropy import units as u


from matplotlib import pyplot as plt
class Potential():
    def __init__(self, machine = 'mac', startsnap = 127, endsnap = 128, loadtypes = [1,2,3,4]):
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
            
        self.startsnap = startsnap
        self.endsnap   = endsnap
        self.loadtypes = loadtypes
        
        for i in range(startsnap, endsnap, 1):
            print("Load snapshot")
            self._load_snapshot(i)
 
    def _load_snapshot(self, snapnr, halo_number = 24, loadtypes = self.loadtypes):
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
            self.s, self.sf = eat_snap_and_fof(level, halo_number, snapnr, snappath, loadonlytype=loadtypes, 
                                               haloid=0, galradfac=0.1, verbose=True) 

            # Clean negative and zero values of gmet to avoid RuntimeErrors
            # later on (e.g. dividing by zero)
            self.s.data['gmet'] = np.maximum( self.s.data['gmet'], 1e-40 )


    def disk_density(self):
        pass
    
    def bulge_density(self):
        pass
    
    def halo_density(self):
        pass
    
    def MNPotential(self):
        
        pass
    
    def HPotential(Self):
        pass
    
    def NFWPotential(self):
        pass
    
       
    def Potential(self):
        disk_pot = self.MNPotential()
        bulge_pot = self.HPotential()
        halo_pot = self.NFWPotential()
        pot = [disk_pot, bulge_pot, halo_pot]
        return(pot)
    
