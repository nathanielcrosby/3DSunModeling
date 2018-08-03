import matplotlib.pyplot as plt
import os
import imageio

def make_movie(files, output, fps=10, **kwargs):
	'''
	uses the imageio library to take the jpegs created before and save them to a string
	of images that is spaced at a certain interval (duration)
	
	Parameters:
	
	files : the array of all the images
	
	output : name of the file outputted
	
	fps : frames per second of the movie
	'''
		
	duration = 1 / fps

	images = []
	for filename in files:
		images.append(imageio.imread(filename))
	#saves array of images as a gif with a set time between each still	
	imageio.mimsave(output, images, duration=duration)
	
def rotanimate(ax, output, azim, elev, fps=10, width=10, height=10, prefix='tmprot_', **kwargs):
    '''
    Produces an animation (.gif) from a 3D plot on a 3D ax
    Makes jpeg pictures of the given 3d ax, with different angles.
    Args:
        ax (3D axis): the ax containing the plot of interest
        output (string): the name of the file created
        azim (list): the list of rotational angles (in degree) under which to show the plot.
        elev (list): the list of elevational angle at which to show the plot
            - width : in inches
            - heigth: in inches
            - fps : frames per second
        prefix (str): prefix for the files created.
        
    Inputs the images, output, and fps to the make_movie function 
    '''   
    
    files = []
    ax.figure.set_size_inches(width, height)
     
    #loops through and creates stills that are added to files array 
    for i,angle in enumerate(azim):
     	#.vew_init sets the view angle based on the list values
        ax.view_init(elev = elev[i], azim=angle)
        fname = '%s%03d.jpeg'%(prefix,i)
        ax.figure.savefig(fname)
        files.append(fname)
           
    make_movie(files, output, fps=fps, **kwargs)
     
    for f in files:
        os.remove(f)