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
from skimage.transform import resize
import subprocess

def km_per_pixel(arcs_per_pix=1.):
	'''
	this uses known values to approximate the number of km per pixel
		
	Parameters:
	
	arcs_per_pix : float, the number of arcseconds per pixel in the image
	
	Returns: km_per_pixel (float)
	'''
	
	dist_earth_to_sun = 151000000. #in km, changes depending on time
	degree_per_arcsec = 1./3600. 
	rad_per_degree = np.pi/180.
	km_per_pixel = (np.sin((degree_per_arcsec * arcs_per_pix * rad_per_degree / 2.))
	 * dist_earth_to_sun * 2.)
	square_km_per_pixel = km_per_pixel**2.
	return km_per_pixel

def stl_flat_maker(data, scale=100, width=100, depth=100, height=20):
	'''
	This uses the stl_tools numpy2stl in order to convert an array into a 3D printable
	model. This cannot take xyz dimensions and cannot make the full 3D model. It makes the
	2D image 3D printable.
	
	Parameters:
	
	data : a numpy array that represents some sort of image to be made into a stl file
	
	scale : float, the percent ratio of the actual height vs the original height. A scale
	of 150 turns a height into 1.5 times more when compared to width than originally.
	 
	width : float, max width in mm
	 
	depth : float, max depth in mm
	 
	height : float, max height in mm
	'''
	
	print('making mesh...')
	
	data = 4*data
	data = imresize(data, (512, 512))
	data = gaussian_filter(data, 1)
	
	numpy2stl(data, '~/Desktop/test.stl', scale=scale, solid=True, max_width=width, 
	max_depth=depth, max_height=height)	

def stl_mesh_maker(x, y, z, interval=1, fname='test1.stl'):
	'''
	This uses the stl mesh.Mesh in order to turn the x, y, and z arrays into a single 
	array with 3 pts each. It then turns it into vectors by taking 2 adjacent points. Used
	to make full 3D models. Interal determines the rate at which points in the array are used
	
	Parameters:
	
	x : x points array
	
	y : y points array
	
	z : z points array
	
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
	subprocess.call(['bash', 'filemover.sh', fname])

def TwoDPlot(image, figx=10., figy=10., save=False, file='2d.png'):
	'''
	2D drawing of the image in a 2D field, literally just shows the image with a heat map
	
	Parameters:
	
	image : numpy array representing the image typically achieved after reading in an
	image with a variety of libraries
	
	figx : float, inches of the x dimension of the plot that appears
	
	figy : float, inches of the y dimension of the plot that appears
	
	save : boolean, whether or not the save the created plot as a png file
	
	file : str, the name of the png file that is potentially saved	
	'''
	
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
	'''
	Used for the 2d image on plane in a 3d diagram, shows the 2D image with a heat map
	at an angle in a 3D plot
	
	Parameters:
	
	image : numpy array representing the image typically achieved after reading in an
	image with a variety of libraries
	
	figx : float, inches of the x dimension of the plot that appears
	
	figy : float, inches of the y dimension of the plot that appears
	
	save : boolean, whether or not the save the created plot as a png file
	
	file : str, the name of the png file that is potentially saved		
	'''
	
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
	'''
	Plots the image in a 3D plot with 3D characteristics. The result with be hemispherical
	and display prominences that are based on the intensity of the pixels.
	
	Parameters:
	
	x : x points array
	
	y : y points array
	
	z : z points array
	
	stride : the interval at which points in the array are taken and plotted. A stride of
	10 means every 10th point from the arrays is plotted.
	
	figx : float, inches of the x dimension of the plot that appears
	
	figy : float, inches of the y dimension of the plot that appears
	
	save : boolean, whether or not the save the created plot as a png file
	
	file : str, the name of the png file that is potentially saved	
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

def make_movie(x, y, z, image, file='movie.gif', fps=30, st_ang=0, en_ang=360, 
st_elev_ang=90, en_elev_ang=0, time=10):
	'''Calls the make movie rotanimate function and makes a move with the specified 
	azim pts, elev pts, filename, and fps
	
	Parameters:
	
	x : x points array
	
	y : y points array
	
	z : z points array
	
	image : the numpy array of the image
	
	file : str, the name of the movie created
	
	fps : int, frames per second of the movie
	
	st_ang : float, initial viewing angle (azim)
	
	en_ang : float end viewing angle (azim)
	
	st_elev_ang : initial elevation angle, straight down is 90, side is 0
	
	en_elev_ang : end elevation angle
	
	time : duration of the film in seconds
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
	'''
	logarithmic scale function, can be used in add_function when creating pts.
	
	Parameters:
	
	intensity : float, the points intensity value
	
	scale_factor : float, the max addition of the prominence
	
	minimum_intensity_threshold : float, the min value the intensity must be to not be 
	set to 0
	
	returns the scaled value (float)
	'''
	return ((scale_factor * ((10.**((intensity - minimum_intensity_threshold) 
	/ (1. - minimum_intensity_threshold))) - 1)) * (1./9.))

def scale(intensity, scale_factor, minimum_intensity_threshold, exp=1):
	'''scale function can be exponential, used in add_function below to make pts.
	
	Parameters:
	
	intensity : the points intensity value
	
	scale_factor : the max addition of the prominence
	
	minimum_intensity_threshold : the min value the intensity must be to not be set to 0
	
	exp : float, the exponent that all points are brought to the power of
	
	returns the scaled value
	'''
	return (scale_factor * ((intensity - minimum_intensity_threshold) 
	/ (1. - minimum_intensity_threshold))**exp)

def init3D_shape(r_len, xDimen, yDimen, xDimen_len, yDimen_len, centerY_len, centerX_len, 
len_per_pixel, scale_factor, minimum_intensity_threshold):
	'''
	This creates the initial 3D shape of the hemisphere that will then be modified with
	intensity values to create the protrusions. This should only be used with the 
	image_to_xyz_mesh, all parameters are the same
	
	Returns the initial hemisphere protruding from a plane in x, y, z array form
	'''
	
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
centerX, centerY, image, scale_bool, r, scale_factor, exp=1., buffer=False, earth=True, border_width = 2, buffer_val=1.1, flat_base=False):
	'''
	This for loop goes through all the points and determines whether or not it is on the
	hemisphere and whether or not the intensity surpasses the threshold. If not, then no 
	multiplier is added. If yes then a multiplier is created based on the intensity of 
	that pixel It also used a minimum intensity threshold to maintain spherical shape and
	Normalizes the values about that threshold. This should only be used with the 
	image_to_xyz_mesh, all parameters are the same.
	
	This returns an array called add that contains the amount the prominences should come out
	'''
	
	print('calculating protrusions...')
	
	sun_radius = 432169.
	earth_radius = 6371. #km
	earth_scale_y = 35
	earth_scale_x = image.shape[1] - 35
	earth_radius_px = (earth_radius / sun_radius) * r
	earth_box = border_width
	
	add = []

	for xpoint in range(xDimen):
		row = []
		for ypoint in range(yDimen):
			
			if ((buffer) and (np.sqrt((xpoint-centerX)**2 + (ypoint-centerY)**2) 
			> (r - buffer_zone))):
				image[xpoint][ypoint]=(float(image[xpoint][ypoint]) / buffer_val)
	
			if (earth):
				if (np.sqrt((xpoint - earth_scale_x)**2 
				+ (ypoint - earth_scale_y)**2) <= earth_radius_px):
					row.append(.1 * scale_factor)
				elif (((np.absolute(xpoint - (earth_scale_x - earth_scale_y)) < earth_box) & (ypoint < (2. * earth_scale_y))) 
				or (((np.absolute(ypoint - (2. * earth_scale_y)) < earth_box)) & (xpoint > (earth_scale_x - earth_scale_y)))):
					row.append(.05 * scale_factor)
			elif ((flat_base) & (np.sqrt((xpoint-centerX)**2 + (ypoint-centerY)**2) 
			> (r))):
				row.append(0.)
			elif ((image[xpoint][ypoint] < minimum_intensity_threshold)):
			 	row.append(0.)			
			elif(scale_bool):
				row.append(scale(image[xpoint][ypoint], scale_factor, 
				minimum_intensity_threshold, exp=exp))
			else:
				row.append(log_scale(image[xpoint][ypoint], scale_factor, 
				minimum_intensity_threshold))
		add.append(row)
	
	return add
	
def final_height_addition(xDimen, yDimen, centerX, centerY, centerX_len, centerY_len, 
r_len, r, x_init, y_init, z_init, add):
	'''
	This goes through all the points and combines the x, y, z arrays with the add array
	in a way that remains normal along the hemisphere and is proportional to the intensity
	of the pixels. This should only be used with the image_to_xyz_mesh, all parameters are
	the same

	This returns the finalized x, y, z arrays
	'''

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
	
def retrieve_image(date, local=True, index=0):
	'''
	This uses the imageFinder.file_finder in order to get an image from the date and 
	then normalize the data
	
	Parameters:
	 
	date : str, the date of the image
	
	local : boolean, if you're in the cfa system and/or have access to 
		/archive/hinode/xrt/level2/synoptics folder
		
	index : int, index of the jp2 image if a specific image is desired, otherwise the 
		first is taken 
	'''
	
	#calling imageFinder program to find the image of the set date
	data, header = imageFinder.file_finder(date, local=local, index=index)
	
	image = data[0][0] #gets the data from the jp2 file, however this format does not work 
	#with the the plot surface, instead it is converted to a png in an oustide program

	image = image.astype(float)

	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			image[x][y] = float(image[x][y]) / 255.
	
	return image, header
	
def image_to_xyz_mesh(date, r=925, base_len=100., offset_x=0, offset_y=0, 
scale_factor_percent=0.25, minimum_intensity_threshold=0.35, buffer_zone=0., buffer_val=1.1,
buffer=False, exp=1., scale_bool=True, earth=True, border_width=2, interval=2., local=True, index=0, flat_base=False):
	'''
	This turns an image from the given date into a 3D printable stl file
	
	Parameters:
	
		date : str, 'YYYY/MM/DD', date of the image
		
		r : float, radius of the sun in px, determines size of hemisphere 
		for 1024x1024 r~460, for 2048x2048 r~925, no need to scale based on interval value		
		
		base_len : float, determines the length of the stl file base in mm
		
		offset_x : int, can be used to offset center of hemisphere to match a not centered
		sun
		
		offset_y : int, same as x, except in y dimension
		
		scale_factor_percent : float, the percent of the radius that the prominences can
		stick out
		
		minimum_intensity_threshold : float (0-1), the percent intensity that the point 
		has to surpass in order to show up as a prominence and not be set to zero
		
		buffer_zone : int, the amount in pixels that within the hemisphere around the outer 
		edge that is set to intensity/1.1 to scale down and act as a buffer to prevent a 
		potentially misshapen or warped hemisphere 
		
		buffer_val : float, the amount that the points in the buffer are divided by
		
		buffer : boolean, determines whether or not the buffer_zone will exist
		
		exp : float, the exponent that all of the intensities are taken to the power of 
		for exponential scaling
		
		scale_bool : boolean, if true, the scaling is exponential (based on exp), if false, 
		the scaling is logarithmic
		
		earth : bool, draw earth to scale in the corner

		border_width : int, width of border around the earth to scale
		
		interval : float, interval at which points are taken from the image
		
		local : boolean, if you're in the cfa system and/or have access to 
		/archive/hinode/xrt/level2/synoptics folder
		
		index : int, index of the jp2 image if a specific image is desired, otherwise the 
		first is taken
		
		flat_base : boolean, whether or not the base should have prominences or just be flat 
	'''
	
	image_raw, header = retrieve_image(date, local=local, index=index)
	
	image = resize(image_raw, (int(image_raw.shape[0] / interval), int(image_raw.shape[1] / interval)))
	
	r = r / interval
	
	buffer_zone = buffer_zone / interval
	
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
	buffer_zone, centerX, centerY, image, scale_bool, r, scale_factor, 
	exp=exp, buffer=buffer, earth=earth, border_width=border_width, buffer_val=buffer_val, flat_base=flat_base)

	x, y, z = final_height_addition(xDimen, yDimen, centerX, centerY, centerX_len, centerY_len, 
	r_len, r, x_init, y_init, z_init, add)
	
	return image_raw, header, x, y, z
	
def image_to_flat_stl(date, scale=100., width=100., depth=100., height=20.):
	'''
	this turns an xrt image from a given date into a flat 3D model so that it has
	 prominences but does not have a hemispherical nature
	 
	 Parameters:
	 
	 date : str, 'YYYY/MM/DD' date that the image is from
	 
	 scale : float, the percent ratio of the actual height vs the original height. A scale
	 of 150 turns a height into 1.5 times more when compared to width than originally.
	 
	 width : float, max width in mm
	 
	 depth : float, max depth in mm
	 
	 height : float, max height in mm
	 '''

	image = retrieve_image(date)
	stl_flat_maker(image, scale=scale, width=width, depth=depth, height=height)
'''
date = '2014/05/16'
r = 460.

image, header, x, y, z = image_to_xyz_mesh(date, r=r, base_len=228.6, offset_x=0., offset_y=0., 
scale_factor_percent=0.25, minimum_intensity_threshold=0.45, buffer_zone=0., buffer_val=1.1,
buffer=True, exp=2.0, scale_bool=True, earth=True, interval=2., local=True, index=2, flat_base=False)

stl_mesh_maker(x, y, z, interval=1, fname='test1.stl')
'''