import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import sunpy.io as io
from matplotlib._png import read_png
from PIL import Image
import urllib
import imageFinder
from matplotlib import animation
import makeMovie
from stl_tools import numpy2stl
from stl import mesh
from scipy.misc import imresize
from scipy.ndimage import gaussian_filter

def km_per_pixel():
	#this uses known values to approximate the number of km per pixel
	dist_earth_to_sun = 148000000 #in km, changes depending on time
	degree_per_arcsec = 1./3600. 
	rad_per_degree = np.pi/180
	km_per_pixel = (np.sin((degree_per_arcsec * rad_per_degree / 2))
	 * dist_earth_to_sun * 2)
	square_km_per_pixel = km_per_pixel**2
	return km_per_pixel

def stl_file_maker(data):
	'''This uses the stl_tools numpy2stl in order to convert an array into a 3D printable
	model. This cannot take xyz dimensions and cannot make the full 3D model. It makes the
	2D image 3D printable.'''
	
	data = 4*data
	data = imresize(data, (512, 512))
	data = gaussian_filter(data, 1)
	
	numpy2stl(data, 'test.stl', scale=scale_factor_percent*1.3, solid=True, max_width=100, 
	max_depth=100, max_height=(r*100/xDimen)*(scale_factor_percent*1.3))	

def stl_mesh_maker(x, y, z, interval=2):
	'''This uses the stl mesh.Mesh in order to turn the x, y, and z arrays into a single 
	array with 3 pts each. It then turns it into vectors by taking 2 adjacent points. Used
	to make full 3D models. Interal determines the rate at which points in the array are used
	'''
	
	base = 5 #mm
	
	for value1 in range(z.shape[0]):
		for value2 in range(z.shape[1]):
			z[value1][value2] += base 
	
	data = []

	k = interval

	for i in range(len(x) / k):
		temp = []
		for j in range(len(x) / k):
			temp.append([x[i*k][j*k], y[i*k][j*k], z[i*k][j*k]])
		data.append(temp)

	data = np.asarray(data)

	vector_data = np.zeros(2*data.shape[0]*data.shape[1] + 10, dtype=mesh.Mesh.dtype)
	
	count = 0
	
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			if (i < data.shape[0] - 1) & (j < data.shape[1] - 1):
				vector_data['vectors'][count] = np.array([data[i][j],
				data[i][j+1], data[i+1][j]])
				count+=1
				if (i > 0) & (j > 0):
					vector_data['vectors'][count] = np.array([data[i][j],
					data[i][j-1], data[i-1][j]])
					count+=1
			
			else:
				vector_data['vectors'][count] = np.array([data[i][j], 
				data[i][j-1], data[i-1][j]])
				count+=1
			
	for value in range(10):
		print(value)
		print(vector_data['vectors'][data.shape[0]*data.shape[1] + value])
		
	print(vector_data['vectors'].shape)
	print(data.shape[0]*data.shape[1])
	print(count)
		
	vector_data['vectors'][data.shape[0]*data.shape[1]*2] = np.array([[0,0,0], 
	[data[data.shape[0]-1][data.shape[1]-1][0],0,0],
	[data[data.shape[0]-1][data.shape[1]-1][0], data[data.shape[0]-1][data.shape[1]-1][1], 0]])
	
	vector_data['vectors'][data.shape[0]*data.shape[1]*2 + 1] = np.array([[0,0,0], 
	[0,data[data.shape[0]-1][data.shape[1]-1][0],0],
	[data[data.shape[0]-1][data.shape[1]-1][0] ,data[data.shape[0]-1][data.shape[1]-1][1] , 0]])
	
	vector_data['vectors'][data.shape[0]*data.shape[1]*2 + 2] = np.array([[0,0,0], 
	[data[data.shape[0]-1][data.shape[1]-1][0],0,0],
	[data[data.shape[0]-1][data.shape[1]-1][0], 0, data[data.shape[0]-1][data.shape[1]-1][2]]])
	
	vector_data['vectors'][data.shape[0]*data.shape[1]*2 + 3] = np.array([[0,0,0], 
	[0,0,data[data.shape[0]-1][data.shape[1]-1][2]],
	[data[data.shape[0]-1][data.shape[1]-1][0], 0, data[data.shape[0]-1][data.shape[1]-1][2]]])
	
	vector_data['vectors'][data.shape[0]*data.shape[1]*2 + 4] = np.array([[0,0,0], 
	[0, data[data.shape[0]-1][data.shape[1]-1][1], 0],
	[0, data[data.shape[0]-1][data.shape[1]-1][1], data[data.shape[0]-1][data.shape[1]-1][2]]])
	
	vector_data['vectors'][data.shape[0]*data.shape[1]*2 + 5] = np.array([[0,0,0], 
	[0, 0, data[data.shape[0]-1][data.shape[1]-1][2]],
	[0, data[data.shape[0]-1][data.shape[1]-1][1], data[data.shape[0]-1][data.shape[1]-1][2]]])
	
	vector_data['vectors'][data.shape[0]*data.shape[1]*2 + 6] = np.array([[0,data[data.shape[0]-1][data.shape[1]-1][1],0], 
	[data[data.shape[0]-1][data.shape[1]-1][0],data[data.shape[0]-1][data.shape[1]-1][1], 0],
	[data[data.shape[0]-1][data.shape[1]-1][0], data[data.shape[0]-1][data.shape[1]-1][1], 
	data[data.shape[0]-1][data.shape[1]-1][2]]])
	
	vector_data['vectors'][data.shape[0]*data.shape[1]*2 + 7] = np.array([[0,data[data.shape[0]-1][data.shape[1]-1][1],0], 
	[0, data[data.shape[0]-1][data.shape[1]-1][1], data[data.shape[0]-1][data.shape[1]-1][2]],
	[data[data.shape[0]-1][data.shape[1]-1][0], data[data.shape[0]-1][data.shape[1]-1][1], 
	data[data.shape[0]-1][data.shape[1]-1][2]]])
	
	vector_data['vectors'][data.shape[0]*data.shape[1]*2 + 8] = np.array([[data[data.shape[0]-1][data.shape[1]-1][0],0,0], 
	[data[data.shape[0]-1][data.shape[1]-1][0], data[data.shape[0]-1][data.shape[1]-1][1], 0],
	[data[data.shape[0]-1][data.shape[1]-1][0], data[data.shape[0]-1][data.shape[1]-1][1], 
	data[data.shape[0]-1][data.shape[1]-1][2]]])
	
	vector_data['vectors'][data.shape[0]*data.shape[1]*2 + 9] = np.array([[data[data.shape[0]-1][data.shape[1]-1][0],0,0], 
	[data[data.shape[0]-1][data.shape[1]-1][0], 0, data[data.shape[0]-1][data.shape[1]-1][2]],
	[data[data.shape[0]-1][data.shape[1]-1][0], data[data.shape[0]-1][data.shape[1]-1][1], 
	data[data.shape[0]-1][data.shape[1]-1][2]]])
	
	new_mesh = mesh.Mesh(vector_data)
	new_mesh.save('test1.stl')
	return new_mesh

def TwoDPlot(fig):
	#2D drawing of the image
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(image)
	ax.imshow(image, cmap=plt.cm.gist_heat, origin='lower')
	#plt.savefig('2d.png')
	plt.show()

def TwoDin3DPlot(fig):
	#Used for the 2d image on plane in a 3d diagram
	x, y = np.mgrid[0:xDimen, 0:yDimen]
	z = np.zeros((xDimen, yDimen))
	
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(x, y, z, rstride=10, cstride=10, antialiased=True, cmap=plt.cm.jet,
	facecolors=plt.cm.jet(image))
	plt.show()

def ThreeDPlot(fig, x, y, z):
	'''Plots the axis created above and allows for specification of initial angle, points,
	color map etc.
	'''
	
	ax = fig.add_subplot(111, projection='3d')

	#uniform scaling of axes so that hemisphere is not stretched
	ax.set_xlim3d(0, 1500000)
	ax.set_ylim3d(0, 1500000)
	ax.set_zlim3d(0, 1500000)

	#labels axes
	ax.set_xlabel('km')
	ax.set_ylabel('km')
	ax.set_zlabel('km')

	#rstride and cstride determine how frequently values are taken from the arrays and 
		#plotted, lower stride yields higher resolution
	#cmap is a color map for the surface patches. This line isn't necessary but makes the 
		#colors match better
	#facecolors sets the image to be drawn and plt.cm.jet normalizes the colors.
		#They then match up with the cmap inputted which is the same
	#antialiased determines whether or not the figure is drawn with antialiasing
	#vmin and vmax determine the range of the colormap: they're not necessary

	ax.plot_surface(x, y, z, rstride=10, cstride=10, antialiased=True, cmap=plt.cm.jet,
	facecolors=plt.cm.jet(image))#, vmin=0., vmax=3000.)
	#plt.cm.jet uses a different color map with a full spectrum... gist_heat... hot

	ax.view_init(elev=45, azim=45)
	
	#plt.savefig('3d.png')
	plt.show()

def make_movie(x, y, z):
	'''Calls the make movie rotanimate function and makes a move with the specified 
	azim pts, elev pts, filename, and fps
	'''
	ax = fig.add_subplot(111, projection='3d')

	#uniform scaling of axes so that hemisphere is not stretched
	ax.set_xlim3d(0, 1500000)
	ax.set_ylim3d(0, 1500000)
	ax.set_zlim3d(0, 1500000)
	
	#no visible axis, good for movie
	plt.axis('off')
	
	ax.plot_surface(x, y, z, rstride=5, cstride=5, antialiased=True, cmap=plt.cm.jet,
	facecolors=plt.cm.jet(image))
	
	azim = np.linspace(0,360,300) # A list of angles between 0 and 360 rotation angle
	elev = np.linspace(90,0,300) # A list of angles between 90 and 0 elevation angle
	# create a movie with 10 frames per seconds and 'quality' 2000
	file = 'movie.gif' #name of movie
	#calling function to make a movie with set points
	makeMovie.rotanimate(ax, file, azim, elev, fps=30)

def log_scale(intensity):
	#logarithmic scale function, can be used below when creating pts.
	return ((scale_factor * ((10.**((intensity - minimum_intensity_threshold) 
	/ (1. - minimum_intensity_threshold))) - 1)) * (1./9.))

def scale(intensity, exp=1):
	#scale function can be exponential, used below to make pts.
	return (scale_factor * ((intensity - minimum_intensity_threshold) 
	/ (1. - minimum_intensity_threshold))**exp)

def init3D_shape():
	#x and y are arrays shaped the size of the pixel dimensions respectively
		#they go from 0 to the dimension in km at intervals of the dimension in km/dimen in px
		#This way each point lines up exactly to a pixel
	x_init, y_init = np.mgrid[0:xDimen_len:(xDimen_len/xDimen), 0:yDimen_len:(yDimen_len/yDimen)]

	#This for loop uses the formula of a sphere (x**2 + y**2 + z**2 = r**2) in order to define
		#z points that are 0 outside the radius and on the hemisphere inside the radius
		#this allows for the hemisphere of a certain radius coming out of the plane
	zlist = []

	for xpoint in range(xDimen):
		xrow = []
		for ypoint in range(yDimen):
			if(np.sqrt((xpoint * len_per_pixel - (centerX_len))**2. + (ypoint * len_per_pixel
			 - (centerY_len))**2.) >= r_len):
				xrow.append(0)
			else:
				xrow.append(np.sqrt(r_len**2. - (xpoint * len_per_pixel - (centerX_len))**2.
				 - (ypoint * len_per_pixel - (centerY_len))**2.))
		zlist.append(xrow)

	#plot surface requires an array
	z_init = np.asarray(zlist)
	return x_init, y_init, z_init
	
def add_function(exp=1., buffer=False, len_per_pixel=km_per_pixel):
	'''This for loop goes through all the points and determines whether or not it is on the
	hemisphere and whether or not the intensity surpasses the threshold. If not, then no 
	multiplier is added. If yes then a multiplier is created based on the intensity of 
	that pixel It also used a minimum intensity threshold to maintain spherical shape and
	Normalizes the values about that threshold'''
	
	add = []

	for xpoint in range(xDimen):
		row = []
		for ypoint in range(yDimen):
			 if ((image[xpoint][ypoint] < minimum_intensity_threshold) 
			 or ((buffer==True) & (np.sqrt((xpoint-centerX)**2 + (ypoint-centerY)**2) < r) 
			 & ((np.sqrt((xpoint-centerX)**2 + (ypoint-centerY)**2) 
			 > (r - buffer_zone))))):
				row.append(0)
			 else:
				row.append(scale(image[xpoint][ypoint], exp=exp))
		add.append(row)
	
	return add
	
def final_height_addition():
	#This for loop goes through the initial x, y, and z values and adds to their position 
		#based on the multipliers intensity. Only points on the hemisphere are added to
		#x, y, and z are all added to so that the bright points go out of the sphere instead
		#of just up which would be the case if just z was added to
	x_list_final = []
	y_list_final = []
	z_list_final = []

	for xpoint in range(xDimen):
		xrow = []
		yrow = []
		zrow = []
		for ypoint in range(yDimen):
			if ((np.sqrt((xpoint-centerX)**2 + (ypoint-centerY)**2)) < r):
				xrow.append(x_init[xpoint][ypoint] + ((x_init[xpoint][ypoint] - (centerX_len))
				 * (add[xpoint][ypoint] / r_len)))
				yrow.append(y_init[xpoint][ypoint] + ((y_init[xpoint][ypoint] - (centerY_len))
				 * (add[xpoint][ypoint] / r_len)))
			else:
				xrow.append(x_init[xpoint][ypoint])
				yrow.append(y_init[xpoint][ypoint])
			zrow.append(z_init[xpoint][ypoint] + add[xpoint][ypoint])
		x_list_final.append(xrow)
		y_list_final.append(yrow)
		z_list_final.append(zrow)

	#plot surface requires an array
	x = np.asarray(x_list_final)
	y = np.asarray(y_list_final)
	z = np.asarray(z_list_final)
	
	return x, y, z


#km_per_pixel = km_per_pixel()
len_per_pixel = 0.05

date = '2007/02/02'
'''
#calling imageFinder program to find the image of the set date
data, header = imageFinder.file_finder(date)

#info from the header
XCEN = header[0].__getitem__('XCEN')
YCEN = header[0].__getitem__('YCEN')
DATEOBS = header[0].__getitem__('CTIME')
NAXIS1 = header[0].__getitem__('NAXIS1')
NAXIS2 = header[0].__getitem__('NAXIS2')
'''
#image = data[0][0] #gets the data from the jp2 file, however this format does not work 
	#with the the plot surface, instead it is converted to a png in an oustide program

#image data is already normalized to [0.0, 1.0]
image = read_png('2007_02_02__12_02_58_608__HINODE_XRT_COMP.png')

#pixel dimensions of the image
xDimen = image.shape[0]
yDimen = image.shape[1]

#Can be used to offset the radius if sun is not in center of image
offsetX = 30. #px
offsetY = -30. #px

#pixel dimensions in kilometers
xDimen_len = xDimen * len_per_pixel
yDimen_len = yDimen * len_per_pixel

#new calculated center of sun
centerX = xDimen/2. - offsetX
centerY = yDimen/2. - offsetY

 #in km
centerX_len = centerX * len_per_pixel
centerY_len = centerY * len_per_pixel

#creates figure
fig = plt.figure(figsize=(10.,10.))

r = 925. #determines the size of the sphere, should be radius of the sun in image
r_len = r * len_per_pixel #in km

x_init, y_init, z_init = init3D_shape()

#Bright features stand out

#how much they could possibly stand out by
scale_factor_percent = 0.2
scale_factor = scale_factor_percent * r_len

minimum_intensity_threshold = 0.3 #intensity values must exceed this in order to 
#become protrusions. This prevents inflation to maintain spherical shape and is not
#necessary for logarithmic and exponential scales
buffer_zone = 25. #region around outside that has no protrusions to
#prevent warping and inflating around edges

add = add_function(exp=1.7, buffer=False)

x, y, z = final_height_addition()

stl_mesh_maker(x, y, z, interval=8)

#ThreeDPlot(fig, x, y, z)

#stl_file_maker(image)

plt.close('all')
