from stl_tools import numpy2stl
from matplotlib._png import read_png
from scipy.misc import imresize
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt

def km_per_pixel(arcs_per_pix = 1.):
	#this uses known values to approximate the number of km per pixel
	dist_earth_to_sun = 151000000. #in km, changes depending on time
	degree_per_arcsec = 1./3600. 
	rad_per_degree = np.pi/180.
	km_per_pixel = (np.sin((degree_per_arcsec * arcs_per_pix * rad_per_degree / 2.))
	 * dist_earth_to_sun * 2.)
	square_km_per_pixel = km_per_pixel**2.
	return km_per_pixel
	
def stl_file_maker(data):
	'''This uses the stl_tools numpy2stl in order to convert an array into a 3D printable
	model. This cannot take xyz dimensions and cannot make the full 3D model. It makes the
	2D image 3D printable.'''
	
	data = data
	#data = gaussian_filter(data, 2)
	
	data =  data[:, :, 0] + 1.*data[:,:, 1]
	data = imresize(data, (data.shape[0], data.shape[1])) 
	#data = gaussian_filter(data, 5)
	
	numpy2stl(data, 'test.stl', scale=.2, solid=True, max_width=100, 
	max_depth=100, max_height=1, force_python=True)	
	
def TwoDPlot(fig):
	#2D drawing of the image
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(image)
	ax.imshow(image)#, cmap=plt.cm.gist_heat, origin='lower')
	#plt.savefig('2d.png')
	plt.show()

scale_factor_percent = 0.3

km_per_pixel = km_per_pixel(0.61)
image = read_png('2014_05_27_14_38_35_AIA_171__AIA_304-2.png')

#creates figure
fig = plt.figure(figsize=(10.,10.))

#TwoDPlot(fig)

stl_file_maker(image)

#closes all plots
plt.close('all')
