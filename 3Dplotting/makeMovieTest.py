import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import os, sys
import numpy as np
import datetime
import imageio
from pprint import pprint
import time
e=sys.exit

def make_movie(files, output, fps=10, **kwargs):
	'''uses the imageio library to take the jpegs created before and save them to a string
	of images that is spaced at a certain interval (duration)'''
		
	duration = 1 / fps

	images = []
	for filename in files:
		images.append(imageio.imread(filename))
	output_file = output
	imageio.mimsave(output_file, images, duration=duration)
	
def rotanimate(ax, azim, output, elev=None, fps=10, width=10, height=10, prefix='tmprot_', **kwargs):
    """
    Produces an animation (.mp4) from a 3D plot on a 3D ax
    Makes jpeg pictures of the given 3d ax, with different angles.
    Args:
        ax (3D axis): the ax containing the plot of interest
        angles (list): the list of angles (in degree) under which to
                       show the plot.
            - width : in inches
            - heigth: in inches
            - framerate : frames per second
            - delay : delay between frames in milliseconds
        ax (3D axis): the ax
        angles (list): the list of angles (in degree) under which to take the picture.
        width,height (float): size, in inches, of the output images.
        prefix (str): prefix for the files created. 
    output: the list of files created (for later removal)
    """
     
    fps = fps
    azimAngle = azim
    prefix = prefix  
    files =  files = []
    ax.figure.set_size_inches(width, height)
     
    for i,angle in enumerate(azimAngle):
     
        ax.view_init(elev = , azim=angle)
        fname = '%s%03d.jpeg'%(prefix,i)
        ax.figure.savefig(fname)
        files.append(fname)
           
    make_movie(files, output, fps=fps, **kwargs)
     
    for f in files:
        os.remove(f)
        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
s = ax.plot_surface(X, Y, Z, cmap=cm.jet)
plt.axis('off') # remove axes for visual appeal
     
angles = np.linspace(0,360,21)[:-1] # Take 20 angles between 0 and 360
 
# create an animated gif (20ms between frames)
rotanimate(ax, angles,'movie.gif',delay=20)