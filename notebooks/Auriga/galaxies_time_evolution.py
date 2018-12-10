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
%matplotlib inline




    
class snap():
    def __init__(self, snapnr, halo_number, machine = 'magny', level = 4):
        if machine == 'magny':
            self.filedir = "/home/extmilan/masterthesis/files/"
            self.basedir = "/hits/universe/GigaGalaxy/level4_MHD/"
            self.plotdir = "/home/extmilan/masterthesis/plots/"
        elif machine == 'mac':
            self.filedir = "/Users/smilanov/Documents/masterthesis/auriga_files/files/"
            self.basedir = "/Users/smilanov/Desktop/Auriga/level4/"
            self.plotdir = "/Users/smilanov/Documents/masterthesis/auriga_files/plots/"
        else:
            raise NotADirectoryError
        halodir = self.basedir + "halo_{0}/".format(halo_number)
        snappath = halodir + "output/"
        
        self.snapnr = snapnr 
        self.halonr = halo_number
        print("halo    : {0}".format(halo_number))
        print("snapnr  : {0}".format(snapnr))
        self.s, self.sf = eat_snap_and_fof(level, halo_number, snapnr, snappath, loadonlytype=[4], 
            haloid=0, galradfac=0.1, verbose=True) 
        
        # Clean negative and zero values of gmet to avoid RuntimeErrors
        # later on (e.g. dividing by zero)
        self.s.data['gmet'] = np.maximum( self.s.data['gmet'], 1e-40 )
        self.min_max_load = False

    def load_min_max_vals(self): 
        if self.min_max_load == False:
            IDs, xmins, xmaxs, ymins, ymaxs, zmins, zmaxs, Rmins, Rmaxs = np.loadtxt('../files/min_max_vals_for_animation_frame_all_simulations.txt')
            haloindex = np.where(IDs == self.halonr)[0][0]
            xmin, xmax = xmins[haloindex], xmaxs[haloindex] 
            ymin, ymax = ymins[haloindex], ymaxs[haloindex] 
            zmin, zmax = zmins[haloindex], zmaxs[haloindex] 
            Rmin, Rmax = Rmins[haloindex], Rmaxs[haloindex] 
            self.xminlim, self.xmaxlim = int(xmin)-10, int(xmax)+10 
            self.yminlim, self.ymaxlim = int(ymin)-10, int(ymax)+10 
            self.zminlim, self.zmaxlim = int(zmin)-10, int(zmax)+10
            self.Rminlim, self.Rmaxlim = int(Rmin)-10, int(Rmax)+10
            self.min_max_load = True
      
    def get_pos_vel(self):
        istars, = np.where( (self.s.type == 4) & (self.s.halo == 0) )#& (s.subhalo == 0))
        (x_kpc, y_kpc, z_kpc), (vx_kms, vy_kms, vz_kms), rxyz_kpc, rxy_kpc = get_cartesian_vectors(self.s, self.sf, istars)
        return((x_kpc, y_kpc, z_kpc), (vx_kms, vy_kms, vz_kms), rxyz_kpc, rxy_kpc)
        
    def get_lookback_time(self):
        snap_time_Gyr = self.s.cosmology_get_lookback_time_from_a( self.s.time, is_flat=True )
        return(snap_time_Gyr)
    
    def plot_data(self):
        (x, y, z), (vx, vy, vz), rxyz, rxy = self.get_pos_vel()
        #, snap_time = read_snap(s_snap[0])
        X = x
        Y = y
        X = np.append(X, [xminlim, xmaxlim])
        Y = np.append(Y, [yminlim, ymaxlim])


cmap = copy.copy(plt.cm.inferno)
cmap.set_bad((0,0,0))  # Fill background with black
norm=matplotlib.colors.LogNorm()
s_snap = np.arange(startnr,endnr,1)


xminlim, xmaxlim, yminlim, ymaxlim = int(xmin)-10, int(xmax)+10, int(ymin)-10, int(ymax)+10 
zminlim, zmaxlim, Rminlim, Rmaxlim = int(zmin)-10, int(zmax)+10, int(Rmin)-10, int(Rmax)+10
### for xy plot ###
(x, y, z), (vx, vy, vz), rxyz, rxy, snap_time = read_snap(s_snap[0])
X = x
Y = y
X = np.append(X, [xminlim, xmaxlim])
Y = np.append(Y, [yminlim, ymaxlim])

fig, ax = plt.subplots(figsize = (8,8))
ax.set_ylim([-400, 400])
ax.set_xlim([-400, 400])
#Create 2d Histogram
data,xedges,yedges = np.histogram2d(X,Y, bins = 501)
#data_bg,x_bg,y_bg = np.histogram2d(X_bg,Y_bg, bins = 501)


#Smooth with filter
ext = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
print(ext)
im = plt.imshow(data.T, origin = 'lower', interpolation = 'gaussian', cmap = cmap, norm=norm, extent = ext, animated=True, )
ax.set_xlabel('x [kpc]')
ax.set_ylabel('y [kpc]')
ax.set_aspect('equal')
time_text = ax.text(0.1, 0.9,'', color = 'orange', transform=ax.transAxes, fontsize=16)

#Define animation. 
def animate(i) :
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
    

anim_xy = animation.FuncAnimation(fig, animate, frames=len(s_snap))

plt.show()

anim_xy.save('../plots/xy_evolution.gif',writer='imagemagick',fps=2)



def read_snap(snapnr, halo_number = 24):
    halodir = basedir+"halo_{0}/".format(halo_number)
    snappath = halodir+"output/"
    
    #print("level   : {0}".format(level))
    #print("halo    : {0}".format(halo_number))
    print("snapnr  : {0}".format(snapnr))
    #print("basedir : {0}".format(basedir))
    #print("halodir : {0}".format(halodir))
    #print("snappath: {0}\n".format(snappath))
    s, sf = eat_snap_and_fof(level, halo_number, snapnr, snappath, loadonlytype=[4], 
        haloid=0, galradfac=0.1, verbose=True) 

    # Clean negative and zero values of gmet to avoid RuntimeErrors
    # later on (e.g. dividing by zero)
    s.data['gmet'] = np.maximum( s.data['gmet'], 1e-40 )


    istars, = np.where( (s.type == 4) & (s.halo == 0) )#& (s.subhalo == 0))
    (x, y, z), (vx, vy, vz), rxyz, rxy = get_cartesian_vectors(s, sf, istars)
    snap_time = s.cosmology_get_lookback_time_from_a( s.time, is_flat=True )
    return(x, y, z), (vx, vy, vz), rxyz, rxy, snap_time