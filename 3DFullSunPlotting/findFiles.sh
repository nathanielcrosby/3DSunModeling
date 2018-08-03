#This uses the wget function to access the web page. 
#-r means it does so recursively so that it gets all the files in the directory
#-np means no parent, so that it does not ascend to the parent directory and download all the images
#-A means accept list
#3 inputs from imageFinder.py are given that specify the date and name of the image
wget -r -np -A .jp2 http://solar.physics.montana.edu/HINODE/XRT/SCIA/synop_official/$1/$2/$3/
