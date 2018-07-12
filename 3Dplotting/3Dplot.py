import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import sunpy.io as io
from matplotlib._png import read_png
from PIL import Image

data = io.read_file('2007_02_02__12_02_58_608__HINODE_XRT_COMP.jp2', filetype='jp2')
header = io.read_file_header('2007_02_02__12_02_58_608__HINODE_XRT_COMP.jp2')

dist_earth_to_sun = 148000000 #in km, changes depending on time
degree_per_arcsec = 1./3600. 
rad_per_degree = np.pi/180
km_per_pixel = (np.sin((degree_per_arcsec * rad_per_degree / 2))
 * dist_earth_to_sun * 2)
square_km_per_pixel = km_per_pixel**2

#info from the header
XCEN = header[0].__getitem__('XCEN')
YCEN = header[0].__getitem__('YCEN')
DATEOBS = header[0].__getitem__('CTIME')
NAXIS1 = header[0].__getitem__('NAXIS1')
NAXIS2 = header[0].__getitem__('NAXIS2')

#image = data[0][0] #gets the data from the jp2 file, however this format does not work 
	#with the the plot surface, instead it is converted to a png in an oustide program

#image data is already normalized to [0.0, 1.0]
image = read_png('2007_02_02__12_02_58_608__HINODE_XRT_COMP.png')

#pixel dimensions of the image
xDimen = image.shape[0]
yDimen = image.shape[1]

#pixel dimensions in kilometers
xDimen_km = xDimen * km_per_pixel
yDimen_km = yDimen * km_per_pixel

#Used for the 2d plane in a 3d diagram
#x, y = np.mgrid[0:xDimen, 0:yDimen]
#z = np.zeros((xDimen, yDimen))

r = 900. #determines the size of the sphere, should be radius of the sun in image
r_km = r * km_per_pixel #in km

#x and y are arrays shaped the size of the pixel dimensions respectively
	#they go from 0 to the dimension in km at intervals of the dimension in km/dimen in px
	#This way each point lines up exactly to a pixel
x_init, y_init = np.mgrid[0:xDimen_km:(xDimen_km/xDimen), 0:yDimen_km:(yDimen_km/yDimen)]

#This for loop uses the formula of a sphere (x**2 + y**2 + z**2 = r**2) in order to define
	#z points that are 0 outside the radius and on the hemisphere inside the radius
	#this allows for the hemisphere of a certain radius coming out of the plane
zlist = []

for xpoint in range(int(xDimen)):
	xrow = []
	for ypoint in range(int(yDimen)):
		if(np.sqrt((xpoint * km_per_pixel - (xDimen_km/2.))**2. + (ypoint * km_per_pixel
		 - (yDimen_km/2.))**2.) >= r_km):
			xrow.append(0)
		else:
			xrow.append(np.sqrt(r_km**2. - (xpoint * km_per_pixel - (xDimen_km/2.))**2.
			 - (ypoint * km_per_pixel - (yDimen_km/2.))**2.))
	zlist.append(xrow)

#plot surface requires an array
z_init = np.asarray(zlist)

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

#uniform scaling
ax2.set_xlim3d(0, 1500000)
ax2.set_ylim3d(0, 1500000)
ax2.set_zlim3d(0, 1500000)

ax2.set_xlabel('km')
ax2.set_ylabel('km')
ax2.set_zlabel('km')

#Bright features stand out
scale_factor = 0.30 * r_km

add = []

for xpoint in range(xDimen):
	row = []
	for ypoint in range(xDimen):
		 if(np.sqrt((xpoint * km_per_pixel - (xDimen_km/2.))**2. + (ypoint * km_per_pixel 
		 - (yDimen_km/2.))**2.) >= r_km):
		 	row.append(0)
		 else:
			row.append(scale_factor * image[xpoint][ypoint])
	add.append(row)

x_list_final = []
y_list_final = []
z_list_final = []

for xpoint in range(xDimen):
	xrow = []
	yrow = []
	zrow = []
	for ypoint in range(yDimen):
		xrow.append(x_init[xpoint][ypoint] + ((x_init[xpoint][ypoint] - (xDimen_km / 2))
		 * (add[xpoint][ypoint] / xDimen_km)))
		yrow.append(y_init[xpoint][ypoint] + ((y_init[xpoint][ypoint] - (yDimen_km / 2))
		 * (add[xpoint][ypoint] / yDimen_km)))
		zrow.append(z_init[xpoint][ypoint] + add[xpoint][ypoint])
	x_list_final.append(xrow)
	y_list_final.append(yrow)
	z_list_final.append(zrow)

x = np.asarray(x_list_final)
y = np.asarray(y_list_final)
z = np.asarray(z_list_final)

#rstride and cstride determine how frequently values are taken from the arrays and 
	#plotted, lower stride yields higher resolution
#norm is the instance of the Normalize class that is used to map the inputted values to
	#actual colors, this input is not needed
#cmap is a color map for the surface patches. This line isn't necessary but makes the 
	#colors match better
#facecolors sets the image to be drawn and plt.cm.jet normalizes the colors.
	#They then match up with the cmap inputted which is the same
#antialiased determines whether or not the figure is drawn with antialiasing
#vmin and vmax determine the range of the colormap: they're not necessary

ax2.plot_surface(x, y, z, rstride=10, cstride=10, antialiased=True, cmap=plt.cm.jet,
facecolors=plt.cm.jet(image))#, vmin=0., vmax=3000.)#, norm=norm)
#plt.cm.gist_heat uses a different color map with a heat spectrum
plt.show()
plt.savefig('3d.png')