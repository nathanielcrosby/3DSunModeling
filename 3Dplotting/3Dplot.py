import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import sunpy.io as io
from matplotlib._png import read_png
from PIL import Image

data = io.read_file('2007_02_02__12_02_58_608__HINODE_XRT_COMP.jp2', filetype='jp2')
header = io.read_file_header('2007_02_02__12_02_58_608__HINODE_XRT_COMP.jp2')

headerList = []

dist_earth_to_sun = 148000000 #in km, changes depending on time
degree_per_arcsec = 1./3600. 
rad_per_degree = np.pi/180
square_km_per_pixel = (np.sin((degree_per_arcsec * rad_per_degree / 2))
 * dist_earth_to_sun * 2)**2

XCEN = header[0].__getitem__('XCEN')
YCEN = header[0].__getitem__('YCEN')
DATEOBS = header[0].__getitem__('CTIME')
NAXIS1 = header[0].__getitem__('NAXIS1')
NAXIS2 = header[0].__getitem__('NAXIS2')

#image = data[0][0]
image = read_png('2007_02_02__12_02_58_608__HINODE_XRT_COMP.png')

#x, y = np.mgrid[0:image.shape[0], 0:image.shape[1]]
#z = np.zeros((image.shape[0], image.shape[1]))

r = 900        #determines the size of the sphere
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0:pi/2.:1800j, 0:2.*pi:1800j] #it is pi/2 to create a hemisphere

#creates the 3d hemisphere to serve as the 'canvas' for the image
x = r * sin(phi) * cos(theta)
y = r * sin(phi) * sin(theta)
z = r/2. * cos(phi)

fig = plt.figure(figsize=(10.,10.))

#2D drawing of the image
#ax1 = fig.add_subplot(1, 1, 1)
#ax1.imshow(image)
#ax1.imshow(image, cmap=plt.cm.gist_heat, origin='lower', vmin=0., vmax=300.)
#plt.show()
#plt.savefig('2d.png')

#This has no effect if removed
#The Normalize function that normalizes data into a 0.0 to 1.0 range
#norm = matplotlib.colors.Normalize()

#2D drawing in a 3D plane
ax2 = fig.add_subplot(111, projection='3d')

#rstride and cstride determine how frequently values are taken from the arrays and plotted, lower stride yields higher resolution
#norm is the instance of the Normalize class that is used to map the inputted values to actual colors, this input is not needed
#cmap is a color map for the surface patches. This line isn't necessary but makes the colors better
#facecolors sets the image to be drawn and plt.cm.jet normalizes the colors.
#	They then match up with the cmap inputted which is the same
#antialiased determines whether or not the figure is drawn with antialiasing

ax2.plot_surface(x, y, z, rstride=5, cstride=5, antialiased=True, cmap=plt.cm.jet,
facecolors=plt.cm.jet(image))#, norm = norm)
plt.show()
#plt.savefig('3d.png')