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
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
for imagePath in glob.glob(args["image"] + "/*.jpg"):

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])

	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)
	#for (i, rect) in enumerate(rects):
	#	print(rect)
	count=0
	c=5
	avgthresh=0
# loop over the face detections
	for (i, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		for (x, y) in shape:
			#print(x,y)
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
		print ("x:",distx,"y:",disty,"thre",thresh)
		squ=open('square','r+')
		'''rnd=open('round','r+')
		het=open('heart','r+')
		squ=open('square','r+')
	
		thresh_lg=float(lg.readline())
		thresh_rnd=float(rnd.readline())
		thresh_het=float(het.readline())
		thresh_squ=float(squ.readline())'''
		
		
		avgthresh1=squ.readline()
		counter1=squ.readline()
		avgthresh= float(avgthresh1)
		counter=float(counter1)
		counter =counter+1
		avgthresh=((avgthresh)*(counter-1)+(thresh))/(counter)
		avgthresh=round(avgthresh,2)
		squ.seek(0)
		squ.write(str(avgthresh)+"\n")
		squ.write(str(counter))
		'''rnd.seek(0)
		het.seek(0)
		squ.seek(0)
		avg_hr=(thresh_rnd+thresh_het)/(2.0)
		avg_ls=(thresh_lg+thresh_squ)/(2.0)
	
		total_thresh=(avg_hr+avg_ls)/(2.0)'''
	
		'''print("total_thresh:",total_thresh)
		print("slope1:",slope1)
		if thresh<=total_thresh:
			print("long or square")
			if slope1>=7.395:
				print ("long face")
				avgthresh=float(lg.readline())
				counter=float(lg.readline())
				counter=counter+1
				avgthresh=((avgthresh)*(counter-1)+(thresh))/(counter)
				lg.seek(0)
				lg.write(str(avgthresh)+"\n")
				lg.write(str(counter))
			else:
				print("square face")
				avgthresh=float(squ.readline())
				counter=float(squ.readline())
				counter=counter+1
				avgthresh=((avgthresh)*(counter-1)+(thresh))/(counter)
				squ.seek(0)
				squ.write(str(avgthresh)+"\n")
				squ.write(str(counter))
		if thresh>total_thresh:
			print("round or heart")
			if slope1>=11.75:
				print("heart face")
				avgthresh=float(het.readline())
				counter=float(het.readline())
				counter=counter+1
				avgthresh=((avgthresh)*(counter-1)+(thresh))/(counter)
				het.seek(0)
				het.write(str(avgthresh)+"\n")
				het.write(str(counter))
			else:
				print("round face")
				avgthresh=float(rnd.readline())
				counter=float(rnd.readline())
				counter=counter+1
				avgthresh=((avgthresh)*(counter-1)+(thresh))/(counter)
				rnd.seek(0)
				rnd.write(str(avgthresh)+"\n")
				rnd.write(str(counter))
	
	
	
		#avgthresh1=f.readline()
		#counter1=f.readline()
		#avgthresh= float(avgthresh1)
		#counter=float(counter1)
		#counter =counter+1
		#avgthresh=((avgthresh)*(counter-1)+(thresh))/(counter)
		#avgthresh=round(avgthresh,2)
		#f.seek(0)
		#f.write(str(avgthresh)+"\n") 
		#f.write("\n")
		#f.write(str(counter));'''
	
		# show the output image with the face detections + facial landmarks
	cv2.imshow("Output", image)
	cv2.waitKey(0)
