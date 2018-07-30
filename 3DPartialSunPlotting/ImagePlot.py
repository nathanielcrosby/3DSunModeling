from stl_tools import numpy2stl
from matplotlib._png import read_png
from scipy.misc import imresize
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import sunpy.io as io

def km_per_pixel(arcs_per_pix = 1.):
	#this uses known values to approximate the number of km per pixel
	dist_earth_to_sun = 151000000. #in km, changes depending on time
	degree_per_arcsec = 1./3600. 
	rad_per_degree = np.pi/180.
	km_per_pixel = (np.sin((degree_per_arcsec * arcs_per_pix * rad_per_degree / 2.))
	 * dist_earth_to_sun * 2.)
	square_km_per_pixel = km_per_pixel**2.
	return km_per_pixel
	
def stl_file_maker(data, interval=1, threshold=0.):
	'''This uses the stl_tools numpy2stl in order to convert an array into a 3D printable
	model. This cannot take xyz dimensions and cannot make the full 3D model. It makes the
	2D image 3D printable.'''
	
	earth_radius_px = earth_radius / km_per_pixel
	
	for xpoint in range(data.shape[0]):
		for ypoint in range(data.shape[1]):
			if (np.sqrt((xpoint - earth_scale_x)**2 + (ypoint - earth_scale_y)**2) <= earth_radius_px):
				data[xpoint][ypoint] = 0.1
				data[xpoint][ypoint] = 0.1
				data[xpoint][ypoint] = 0.1
			elif (data[xpoint][ypoint] <= threshold):
				data[xpoint][ypoint] = 0
			else:
				data[xpoint][ypoint] = (data[xpoint][ypoint] - threshold) / (1 - threshold)
	
	#data =  data[:, :, 0] + 1.*data[:,:, 1]
	data = imresize(data, (data.shape[0]/interval, data.shape[1]/interval)) 
	data = gaussian_filter(data, 1)
	
	numpy2stl(data, 'test.stl', scale=.2, solid=True, max_width=100, 
	max_depth=100, max_height=20, force_python=True)	
	
def TwoDPlot(fig):
	#2D drawing of the image
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(image)
	ax.imshow(image)#, cmap=plt.cm.gist_heat, origin='lower')
	#plt.savefig('2d.png')
	plt.show()

earth_radius = 6371 #km

scale_factor_percent = 0.3

km_per_pixel = km_per_pixel(0.61)
#image = read_png('2014_05_27_14_38_35_AIA_171__AIA_304-2.png')

#data = io.read_file('2014_05_27__14_38_31_12__SDO_AIA_AIA_304.jp2')
	
#header = io.read_file_header('2014_05_27__14_38_31_12__SDO_AIA_AIA_304.jp2')

image = read_png('2014_05_27__14_38_31_12__SDO_AIA_AIA_304.png')

earth_scale_x = 35
earth_scale_y = image.shape[1] - 35 #px

#creates figure
fig = plt.figure(figsize=(10.,10.))

#TwoDPlot(fig)

stl_file_maker(image, interval=2)

#closes all plots
plt.close('all')
