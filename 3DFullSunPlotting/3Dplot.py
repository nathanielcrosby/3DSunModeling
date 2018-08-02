import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import sunpy.io as io
from matplotlib._png import read_png
import urllib
import imageFinder
from matplotlib import animation
import makeMovie
from stl_tools import numpy2stl
from stl import mesh
from scipy.misc import imresize
from scipy.ndimage import gaussian_filter

def km_per_pixel(arcs_per_pix = 1.):
	'''this uses known values to approximate the number of km per pixel'''
	
	dist_earth_to_sun = 151000000. #in km, changes depending on time
	degree_per_arcsec = 1./3600. 
	rad_per_degree = np.pi/180.
	km_per_pixel = (np.sin((degree_per_arcsec * arcs_per_pix * rad_per_degree / 2.))
	 * dist_earth_to_sun * 2.)
	square_km_per_pixel = km_per_pixel**2.
	return km_per_pixel

def stl_file_maker(data, scale=100, width=100, depth=100, height=20):
	'''This uses the stl_tools numpy2stl in order to convert an array into a 3D printable
	model. This cannot take xyz dimensions and cannot make the full 3D model. It makes the
	2D image 3D printable.'''
	
	print('making mesh...')
	
	data = 4*data
	data = imresize(data, (512, 512))
	data = gaussian_filter(data, 1)
	
	numpy2stl(data, '~/Desktop/test.stl', scale=scale, solid=True, max_width=width, 
	max_depth=depth, max_height=height)	

def stl_mesh_maker(x, y, z, interval=2, fname='test1.stl'):
	'''
	This uses the stl mesh.Mesh in order to turn the x, y, and z arrays into a single 
	array with 3 pts each. It then turns it into vectors by taking 2 adjacent points. Used
	to make full 3D models. Interal determines the rate at which points in the array are used
	
	Parameters:
	
	x : x array
	
	y : y array
	
	z : z array
	
	interval: int, the rate at which values are parsed from the x, y, z, arrays in the
	creation of the stl file, lower -> higher resolution
		
	fname: str, name of the stl file created
	'''
	
	print('creating mesh...')
	
	#adding a base to the stl file so that every part of the file has a width
	base = 4 #mm
	
	for value1 in range(z.shape[0]):
		for value2 in range(z.shape[1]):
			z[value1][value2] += base 
	
	data = []

	#interval at which the data is taken from the arrays
	k = interval

	for i in range(len(x) / k):
		temp = []
		for j in range(len(x) / k):
			temp.append([x[i*k][j*k], y[i*k][j*k], z[i*k][j*k]])
		data.append(temp)

	data = np.asarray(data)

	#contains all the vectors created from data points
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
		
	print('closing shape...')
	
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
	new_mesh.save(fname)
	return new_mesh

def TwoDPlot(image, figx=10., figy=10., save=False, file='2d.png'):
	'''2D drawing of the image'''
	
	print('2D plotting...')
	
	#creates figure
	fig = plt.figure(figsize=(figx,figy))
	
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(image)
	ax.imshow(image, cmap=plt.cm.hot, origin='lower')
	
	if(save):
		plt.savefig(file)
		
	plt.show()

def TwoDin3DPlot(image, figx=10., figy=10., save=False, file='2din3d.png'):
	'''Used for the 2d image on plane in a 3d diagram'''
	
	print('2D plot in 3D...')
	
	#creates figure
	fig = plt.figure(figsize=(figx,figy))
	
	x, y = np.mgrid[0:xDimen, 0:yDimen]
	z = np.zeros((xDimen, yDimen))
	
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(x, y, z, rstride=10, cstride=10, antialiased=True, cmap=plt.cm.hot,
	facecolors=plt.cm.hot(image))
	
	if(save):
		plt.savefig(file)
	
	plt.show()

def ThreeDPlot(x, y, z, image, stride=10, figx=10., figy=10., save=False, file='3d.png'):
	'''Plots the axis created above and allows for specification of initial angle, points,
	color map etc.
	'''
	
	print('3D Plotting...')

	#creates figure
	fig = plt.figure(figsize=(figx,figy))
	
	ax = fig.add_subplot(111, projection='3d')

	#uniform scaling of axes so that hemisphere is not stretched
	ax.set_xlim3d(0, x.max())
	ax.set_ylim3d(0, x.max())
	ax.set_zlim3d(0, x.max())

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

	ax.plot_surface(x, y, z, rstride=stride, cstride=stride, antialiased=True, cmap=plt.cm.hot,
	facecolors=plt.cm.hot(image))
	#plt.cm.jet uses a different color map with a full spectrum... gist_heat... hot

	ax.view_init(elev=45, azim=45)
	
	if save:
		plt.savefig(file)
		
	plt.show()

def make_movie(x, y, z, image, file='movie.gif', fps=30, st_ang=0, en_ang=360, st_elev_ang=90, en_elev_ang=0, time=10):
	'''Calls the make movie rotanimate function and makes a move with the specified 
	azim pts, elev pts, filename, and fps
	'''
	
	print('making movie...')
	
	ax = fig.add_subplot(111, projection='3d')

	#uniform scaling of axes so that hemisphere is not stretched
	ax.set_xlim3d(0, x.max())
	ax.set_ylim3d(0, x.max())
	ax.set_zlim3d(0, x.max())
	
	#no visible axis, good for movie
	plt.axis('off')
	
	ax.plot_surface(x, y, z, rstride=5, cstride=5, antialiased=True, cmap=plt.cm.hot,
	facecolors=plt.cm.hot(image))
	
	azim = np.linspace(st_ang,en_ang,fps*time) # A list of angles between 0 and 360 rotation angle
	elev = np.linspace(st_elev_ang,en_elev_ang,fps*time) # A list of angles between 90 and 0 elevation angle
	#calling function to make a movie with set points
	makeMovie.rotanimate(ax, file, azim, elev, fps=fps)

def log_scale(intensity, scale_factor, minimum_intensity_threshold):
	#logarithmic scale function, can be used below when creating pts.
	return ((scale_factor * ((10.**((intensity - minimum_intensity_threshold) 
	/ (1. - minimum_intensity_threshold))) - 1)) * (1./9.))

def scale(intensity, scale_factor, minimum_intensity_threshold, exp=1):
	#scale function can be exponential, used below to make pts.
	return (scale_factor * ((intensity - minimum_intensity_threshold) 
	/ (1. - minimum_intensity_threshold))**exp)

def init3D_shape(r_len, xDimen, yDimen, xDimen_len, yDimen_len, centerY_len, centerX_len, 
len_per_pixel, scale_factor, minimum_intensity_threshold):
	'''This creates the initial 3D shape of the hemisphere that will then be modified with
	intensity values to create the protrusions'''
	
	print('creating hemisphere...')

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
	
def add_function(len_per_pixel, xDimen, yDimen, minimum_intensity_threshold, buffer_zone, 
centerX, centerY, image, scale_bool, r, scale_factor, exp=1., buffer=False):
	'''This for loop goes through all the points and determines whether or not it is on the
	hemisphere and whether or not the intensity surpasses the threshold. If not, then no 
	multiplier is added. If yes then a multiplier is created based on the intensity of 
	that pixel It also used a minimum intensity threshold to maintain spherical shape and
	Normalizes the values about that threshold'''
	
	print('calculating protrusions...')
	
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
			 	if(scale_bool):
					row.append(scale(image[xpoint][ypoint], scale_factor, 
					minimum_intensity_threshold, exp=exp))
				else:
					row.append(log_scale(image[xpoint][ypoint], scale_factor, 
					minimum_intensity_threshold))
		add.append(row)
	
	return add
	
def final_height_addition(xDimen, yDimen, centerX, centerY, centerX_len, centerY_len, 
r_len, r, x_init, y_init, z_init, add):

	print('adding protrusions...')

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
	
def retrieve_image(date):
	'''This uses the imageFinder.file_finder in order to get an image from the date and 
	then normalize the data'''
	
	#calling imageFinder program to find the image of the set date
	data, header = imageFinder.file_finder(date)
	
	image = data[0][0] #gets the data from the jp2 file, however this format does not work 
	#with the the plot surface, instead it is converted to a png in an oustide program

	image = image.astype(float)

	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			image[x][y] = float(image[x][y]) / 255.
	
	return image, header
	
def image_to_stl_mesh(date, r=925, base_len=100., offset_x=0, offset_y=0, 
scale_factor_percent=0.25, minimum_intensity_threshold=0.35, buffer_zone=0., buffer=False,
exp=1., scale_bool=True):
	'''
	This turns an image from the given date into a 3D printable stl file
	
	Parameters:
	
		date : str, 'YYYY/MM/DD', date of the image
		
		r : float, radius of the sun in px, determines size of hemisphere
		
		base_len : float, determines the length of the stl file base in mm
		
		offset_x : int, can be used to offset center of hemisphere to match a not centered
		sun
		
		offset_y : int, same as x, except in y dimension
		
		scale_factor_percent : float, the percent of the radius that the prominences can
		stick out
		
		minimum_intensity_threshold : float (0-1), the percent intensity that the point 
		has to surpass in order to show up as a prominence and not be set to zero
		
		buffer_zone : int, the amount in pixels that within the hemisphere around the outer 
		edge that is set to 0 as a buffer to prevent a potentially misshapen or warped
		hemisphere 
		
		buffer : boolean, determines whether or not the buffer_zone will exist
		
		exp : float, the exponent that all of the intensities are taken to the power of 
		for exponential scaling
		
		scale_bool : boolean, if true, the scaling is exponential (based on exp), if false, 
		the scaling is logarithmic
	'''
	
	image, header = retrieve_image(date)
	
	#pixel dimensions of the image
	xDimen = image.shape[0]
	yDimen = image.shape[1]
	
	len_per_pixel = base_len / xDimen
	
	#pixel dimensions in kilometers
	xDimen_len = xDimen * len_per_pixel
	yDimen_len = yDimen * len_per_pixel
	
	#new calculated center of sun
	centerX = xDimen/2. - offset_x
	centerY = yDimen/2. - offset_y

	 #in km
	centerX_len = centerX * len_per_pixel
	centerY_len = centerY * len_per_pixel
	
	r = float(r) #determines the size of the sphere, should be radius of the sun in image
	r_len = r * len_per_pixel #in km

	scale_factor = scale_factor_percent * r_len
	
	x_init, y_init, z_init = init3D_shape(r_len, xDimen, yDimen, xDimen_len, yDimen_len,
	centerY_len, centerX_len, len_per_pixel, scale_factor, minimum_intensity_threshold)

	add = add_function(len_per_pixel, xDimen, yDimen, minimum_intensity_threshold, 
	buffer_zone, centerX, centerY, image, scale_bool, r, scale_factor, exp=exp, buffer=True)

	x, y, z = final_height_addition(xDimen, yDimen, centerX, centerY, centerX_len, centerY_len, 
	r_len, r, x_init, y_init, z_init, add)
	
	return x, y, z

date = '2007/02/02'
r = 925.

image_to_stl_mesh(date, r=r, base_len=100., offset_x=30, offset_y=-30, 
scale_factor_percent=0.25, minimum_intensity_threshold=0.35, buffer_zone=0., buffer=False,
exp=1.3, scale_bool=True)

stl_mesh_maker(x, y, z, interval=4, fname='test1.stl')
