import sunpy.io as io
import subprocess
import os

def file_finder(date, local=True, index=0):
	''' 
	This function takes in the date from the 3Dplot.py. It then uses this date to 
	access the website that contains all of the XRT images. It uses a findFiles.sh bash 
	script that attempts to download images of a certain url. It attempts to match an 
	image with a date and if successful downloads the images for that date. It then turns
	the first image into an array and a header. If it was unsuccessful, it iterates 
	through dates until it finds a day with pictures
	'''

	date = date
	print("initial date is: %s" % date)
	dirname = []
	year_file = date[0:4]
	month_file = date[5:7]
	day_file = date[8:10]

	internet_dir = 'solar.physics.montana.edu/HINODE/XRT/SCIA/synop_official/'
	local_dir = '/archive/hinode/xrt/level2/synoptics/'
	
	if (local):
		dir = local_dir
	else:
		dir = internet_dir

	counter = 0
	while len(dirname) <= 0:
		#calling bash script
		if (not local):
			subprocess.call(['bash', 'findFiles.sh', year_file, month_file, day_file])
		date_file = year_file + '/' + month_file + '/' + day_file + '/'
		#loop through files in the directory of year, month, and day finding all files/dirs
		for dirnames in os.walk(dir+date_file):
			counter += 1
			dirname.append(dirnames)
		
		print(year_file, month_file, day_file)	
		print(counter)
		
		#iterates through dates if initial date is unsuccessful	
		if counter <= 0:
			if((int(month_file) == 2) & (int(day_file) >= 28)):
				day_file = '0' + str(1)
				month_file = '0' + str(3)
			elif(((int(month_file) == 1) or (int(month_file) == 3) or (int(month_file) == 5) 
			or (int(month_file) == 7) or (int(month_file) == 8) or (int(month_file) == 10)) 
			& (int(day_file) >= 31)):
				day_file = '0' + str(1)
				if(int(month_file) < 9):
					month_file = '0' + str(int(month_file) + 1)
				else:
					month_file = str(int(month_file) + 1)
			elif(((int(month_file) == 4) or (int(month_file) == 6) or (int(month_file) == 9) 
			or (int(month_file) == 11)) & (int(day_file) >= 30)):
				day_file = '0' + str(1)
				if(int(month_file) < 9):
					month_file = '0' + str(int(month_file) + 1)
				else:
					month_file = str(int(month_file) + 1)
			elif((int(month_file) == 12) and (int(day_file) >= 31)):
				day_file = '0' + str(1)
				month_file = '0' + str(1)
				year_file = str(int(year_file) + 1)
			else:
				if(int(day_file) < 9):
					day_file = '0' + str(int(day_file) + 1)
				else:
					day_file = str(int(day_file) + 1)
		
	date_file = year_file + '/' + month_file + '/' + day_file + '/'
	print('date of image: %s' % date_file)

	if (local):
		index = 3*index + 2

	#picks first image in first directory
	Hname = str(dirname[0][1][0])
	filename = str(dirname[1][2][index])
	#uses sunpy to read .jp2 file into data and header
	data = io.read_file(dir+date_file+Hname+'/'+filename)
	header = io.read_file_header(dir+date_file+Hname+'/'+filename)
	return data, header
