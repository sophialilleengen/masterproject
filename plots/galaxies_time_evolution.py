import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.colors import LogNorm
import copy

from areposnap.gadget import gadget_readsnap
from areposnap.gadget_subfind import load_subfind

from auriga_basics import *
%matplotlib inlines 


level = 4
machine = 'mac'
machine = 'magny'

if machine == 'magny':
    filedir = "/home/extmilan/masterthesis/files/"
    basedir = "/hits/universe/GigaGalaxy/level4_MHD/"
    plotdir = "/home/extmilan/masterthesis/plots/"
elif machine == 'mac':
    filedir = "/Users/smilanov/Documents/masterthesis/auriga_files/files/"
    basedir = "/Users/smilanov/Desktop/Auriga/level4/"
else:
    raise NotADirectoryError
    
infile_minmax = filedir + 'min_max_vals_for_animation_frame_all_simulations.txt'
halnums, xmins, xmaxs, ymins, ymaxs, zmins, zmaxs, Rmins, Rmaxs = np.loadtxt(infile_minmax, unpack = True)
    
class snapshot():
    def __init__(self, halonumber, snapnumber, get_minmaxvals):
        halodir = basedir+"halo_{0}/".format(halonumber)
        snappath = halodir+"output/"

        #print("level   : {0}".format(level))
        print("halo    : {0}".format(halonumber))
        print("snapnr  : {0}".format(snapnumber))
        #print("basedir : {0}".format(basedir))
        #print("halodir : {0}".format(halodir))
        #print("snappath: {0}\n".format(snappath))
        self.s, self.sf = eat_snap_and_fof(level, halonumber, snapnumber, snappath, loadonlytype=[4], 
            haloid=0, galradfac=0.1, verbose=True) 

        # Clean negative and zero values of gmet to avoid RuntimeErrors
        # later on (e.g. dividing by zero)
        self.s.data['gmet'] = np.maximum( self.s.data['gmet'], 1e-40 )      
        
        ## find min and max values of x, y, z and R in all snapshots to set frame for animation
        self.xmin = xmins[halonumber-1]
        self.xmax = xmaxs[halonumber-1]
        self.ymin = ymins[halonumber-1]
        self.ymax = ymaxs[halonumber-1]
        self.zmin = zmins[halonumber-1]
        self.zmax = zmaxs[halonumber-1]
        self.Rmin = Rmins[halonumber-1]
        self.Rmax = Rmaxs[halonumber-1]
        
        
        self.xminlim, self.xmaxlim, self.yminlim, self.ymaxlim = int(self.xmin)-10, int(self.xmax)+10, int(self.ymin)-10, int(self.ymax)+10 
        self.zminlim, self.zmaxlim, self.Rminlim, self.Rmaxlim = int(self.zmin)-10, int(self.zmax)+10, int(self.Rmin)-10, int(self.Rmax)+10
        
    def get_lookback_time(self):
        snap_time = self.s.cosmology_get_lookback_time_from_a( self.s.time, is_flat=True )
        return(snap_time)

    def step(self, i): 
        istars, = np.where( (self.s.type == 4) & (self.s.halo == 0) )#& (s.subhalo == 0))
        (x, y, z), (vx, vy, vz), rxyz, rxy = get_cartesian_vectors(self.s, self.sf, istars)
        return(x, y, z), (vx, vy, vz), rxyz, rxy
        

def animate(i) :
    snapshot.step
    (x, y, z), (vx, vy, vz), rxyz, rxy, snap_time = read_snap(s_snap[i])
    X = x
    Y = y
    X = np.append(X, [xminlim, xmaxlim])
    Y = np.append(Y, [yminlim, ymaxlim])    
    #X = np.append(X, [-201, 201])
    #Y = np.append(Y, [-201, 201])
    data,xedge,yedge = np.histogram2d(X,Y, bins = 501)
    
    im.set_data(data)
    
    time_text.set_text("Lookback time: {:.2f} Gyr".format(snap_time) )