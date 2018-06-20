# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math
import glob
import os,sys
import operator

from PIL import Image
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
size=500,500
threshold=open('threshold','r+')
face=open('face','r+')
path=open('path','r+')
thresh_1=threshold.read()
face_1=face.read()
path_1=path.read()
threshold1=thresh_1.rsplit(' ',100)
face1=face_1.rsplit(' ',100)
path1=path_1.rsplit(' ',100)
t=0
face_arr=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
thresh_arr=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
path_arr=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
new_arr=[999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999.999,999,999,999,999,999,999,999,999,999.999,999,999,999,999,999,999,999,999,999.999,999,999,999,999,999,999,999,999,999]
index_arr=[999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999]

while t<=59:
	face_arr[t]=face_1.rsplit(' ',100)[t]
	thresh_arr[t]=thresh_1.rsplit(' ',100)[t]
	path_arr[t]=path_1.rsplit(' ',100)[t]
	d=int(face_arr[t])
	t=t+1
'''t=0
while t<=39:
	print(face_arr[t]) 
	print(thresh_arr[t])
	print(path_arr[t])
	t=t+1'''

#for imagePath in glob.glob(args["image"] + "/*.jpg"):
	
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

	
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	
rects = detector(gray, 1)
	
count=0
c=5
counter=0
avgthresh=0
	
for (i, rect) in enumerate(rects):
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	for (x, y) in shape:
			
		cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
	for (x, y) in shape:
		count=count+1;
		if(count==1):
			(x1,y1)=(x,y)
		if(count==3):
			(x3,y3)=(x,y)
		if(count==5):
			(x5,y5)=(x,y)
		if(count==7):
			(x7,y7)=(x,y)
		if(count==9):
			(x9,y9)=(x,y)
		if(count==17):
			(x17,y17)=(x,y)
		if(count==28):
			(x28,y28)=(x,y)
	
	#print("1.",(x1,y1),"3.",(x17,y17),"5.",(x9,y9),"7.",(x28,y28))
	slope1=((y3-y1)*(1.0))/((x3-x1)*(1.0))
	slope2=((y5-y3)*(1.0))/((x5-x3)*(1.0))
	slope3=((y7-y5)*(1.0))/((x7-x5)*(1.0))
	slope4=((y9-y7)*(1.0))/((x9-x7)*(1.0))		
	#print('s1:',slope1,'s2:',slope2,'s3:',slope3,'s4:',slope4)
	
	distx=math.sqrt(pow((x1-x17),2)+pow((y1-y17),2))
	disty=math.sqrt(pow((x9-x28),2)+pow((y9-y28),2))
	thresh=distx-disty
	#threshold.write(str(thresh)+" ")
	#new.write(str(thresh)+"\t")
		
	print ("x:",distx,"y:",disty,"thre",thresh)
	lg=open('long','r+')
	rnd=open('round','r+')
	het=open('heart','r+')
	squ=open('square','r+')
	
	thresh_lg=float(lg.readline())
	thresh_rnd=float(rnd.readline())
	thresh_het=float(het.readline())
	thresh_squ=float(squ.readline())
		
	lg.seek(0)
	rnd.seek(0)
	het.seek(0)
	squ.seek(0)
	avg_hr=(thresh_rnd+thresh_het)/(2.0)
	avg_ls=(thresh_lg+thresh_squ)/(2.0)
	
	total_thresh=(avg_hr+avg_ls)/(2.0)
	
	print("total_thresh:",total_thresh)
	print("slope1:",slope1)
	if thresh<=total_thresh:
		print("long or square")
		if slope1>=7.395:
			if slope3>=1.15:
				print ("long face")
				face1=0
				avgthresh=float(lg.readline())
				counter=float(lg.readline())
				counter=counter+1
				avgthresh=((avgthresh)*(counter-1)+(thresh))/(counter)
				lg.seek(0)
				lg.write(str(avgthresh)+"\n")
				lg.write(str(counter))
				#face.write(str(face1)+" ")
			else:
				print("square face")
				face1=1
				avgthresh=float(squ.readline())
				counter=float(squ.readline())
				counter=counter+1
				avgthresh=((avgthresh)*(counter-1)+(thresh))/(counter)
				squ.seek(0)
				squ.write(str(avgthresh)+"\n")
				squ.write(str(counter))
				#face.write(str(face1)+" ")
		elif slope1<7.395:
			if slope3>=1.15:
				print("square face")
				face1=1
				avgthresh=float(squ.readline())
				counter=float(squ.readline())
				counter=counter+1
				avgthresh=((avgthresh)*(counter-1)+(thresh))/(counter)
				squ.seek(0)
				squ.write(str(avgthresh)+"\n")
				squ.write(str(counter))
				#face.write(str(face1)+" ")
			else:
				print ("long face")
				face1=0
				avgthresh=float(lg.readline())
				counter=float(lg.readline())
				counter=counter+1
				avgthresh=((avgthresh)*(counter-1)+(thresh))/(counter)
				lg.seek(0)
				lg.write(str(avgthresh)+"\n")
				lg.write(str(counter))
				#face.write(str(face1)+" ")
				
	if thresh>total_thresh:
		print("round or heart")
		if slope1>=11.75:
			if slope3<=1.1:
				print("heart face")
				face1=2
				avgthresh=float(het.readline())
				counter=float(het.readline())
				counter=counter+1
				avgthresh=((avgthresh)*(counter-1)+(thresh))/(counter)
				het.seek(0)
				het.write(str(avgthresh)+"\n")
				het.write(str(counter))
				#face.write(str(face1)+" ")
			else:
				print("round face")
				face1=3
				avgthresh=float(rnd.readline())
				counter=float(rnd.readline())
				counter=counter+1
				avgthresh=((avgthresh)*(counter-1)+(thresh))/(counter)
				rnd.seek(0)
				rnd.write(str(avgthresh)+"\n")
				rnd.write(str(counter))
				#face.write(str(face1)+" ")
			
		elif slope1<11.75:
			if slope3>1.1:
				print("round face")
				face1=3
				avgthresh=float(rnd.readline())
				counter=float(rnd.readline())
				counter=counter+1
				avgthresh=((avgthresh)*(counter-1)+(thresh))/(counter)
				rnd.seek(0)
				rnd.write(str(avgthresh)+"\n")
				rnd.write(str(counter))
				#face.write(str(face1)+" ")
			else:
				print("heart face")
				face1=2
				avgthresh=float(het.readline())
				counter=float(het.readline())
				counter=counter+1
				avgthresh=((avgthresh)*(counter-1)+(thresh))/(counter)
				het.seek(0)
				het.write(str(avgthresh)+"\n")
				het.write(str(counter))
				#face.write(str(face1)+" ")
		
			
				
				
	#path.write(str(imagePath)+" ")
		
	#im=Image.open(open(str(imagePath), 'rb'))
	#im=imutils.resize(im, width=500)
	#im.show()
	t=0
	cc=0	
	data=0	
	while t<=99:
		data=int(face_arr[t])
		if data==face1:
			new_arr[cc]=abs(float(thresh_arr[t])-thresh)
			index_arr[cc]=t
			cc=cc+1
		t=t+1
	counter=0
	t=0		
	while t<=99:
		min_index,value=min(enumerate(new_arr),key=operator.itemgetter(1))
		if  counter<=4:
			k=index_arr[min_index]
			new_arr[min_index]=999
			counter=counter+1
			image11=path_arr[k]
			im=Image.open(open(str(image11), 'rb'))
			im.thumbnail(size, Image.ANTIALIAS)
			im.show()
		t=t+1
	'''t=0
	while t<=59:
		print(index_arr[t])
		t=t+1'''
				
	cv2.imshow("Output", image)
	cv2.waitKey(0)
