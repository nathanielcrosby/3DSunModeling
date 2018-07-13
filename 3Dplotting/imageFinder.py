import sunpy.io as io
import subprocess
import os

def file_finder(date):
	''' This function takes in the date from the 3Dplot.py. It then uses this date to access
	the website that contains all of the XRT images. It uses a findFiles.sh bash script that
	attempts to download images of a certain url. It attempts to match an image with a date
	and if successful downloads the images for that date. It then turns the first image into
	an array and a header. If it was unsuccessful, it iterates through dates until it finds
	a day with pictures'''
	date = date
	year = date[0:4]
	month = date[5:7]
	day = date[8:10]
	print(year, month, day)
	dirname = []
	year_file = year
	month_file = month
	day_file = day
	counter = 0
	while len(dirname) <= 0:
		subprocess.call(['bash', 'findFiles.sh', year_file, month_file, day_file])
		date_file = year_file + '/' + month_file + '/' + day_file + '/'
		for dirnames in os.walk('solar.physics.montana.edu/HINODE/XRT/SCIA/synop_official/'+date_file):
			counter += 1
			dirname.append(dirnames)
			
		print(year_file, month_file, day_file)	
		print(counter)
			
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
	Hname = str(dirname[0][1][0])
	filename = str(dirname[1][2][0])
	data = io.read_file('solar.physics.montana.edu/HINODE/XRT/SCIA/synop_official/'
	+date_file+Hname+'/'+filename)
	header = io.read_file_header('solar.physics.montana.edu/HINODE/XRT/SCIA/synop_official/'
	+date_file+Hname+'/'+filename)
	return data, header
