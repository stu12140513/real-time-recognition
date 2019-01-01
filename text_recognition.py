# -*- coding: utf-8 -*-
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
import json
import requests
def decode_predictions(scores, geometry):
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	for y in range(0, numRows):
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		for x in range(0, numCols):
			if scoresData[x] < 0.5:
				continue
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	return (rects, confidences)
def text_recognitioner(IMGfromview,net):
	#image = cv2.imread(args["image"])
	image = IMGfromview
	orig = image.copy()
	(origH, origW) = image.shape[:2]
	(newW, newH) = (320, 320)
	rW = origW / float(newW)
	rH = origH / float(newH)
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]
	layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
	#net = cv2.dnn.readNet('C:/Users/t9601/a/b/EAST/frozen_east_text_detection.pb')
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	(rects, confidences) = decode_predictions(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)
	results = []
	resultstext=[]
	i = 0
	for (startX, startY, endX, endY) in boxes:
		roi = 0
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)
		dX = int((endX - startX) * 0)
		dY = int((endY - startY) * 0)
		startX = max(0, startX - dX)
		startY = max(0, startY - dY)
		endX = min(origW, endX + (dX * 2))
		endY = min(origH, endY + (dY * 2))
		if startY<7:
			roi = orig[startY:endY+6, startX-6:endX+6] #<-------------------------修改部分
		elif startX<7:
			roi = orig[startY-6:endY+6, startX:endX+6]
		elif startY<7 and startX<7:
			roi = orig[startY:endY+6, startX:endX+6]
		else:
			roi = orig[startY-6:endY+6, startX-6:endX+6] #<------------------------到這邊
		#img = cv2.resize(roi, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
		i = i+1
		while roi.shape[0]>40 or roi.shape[1]>150 :
			print(roi.shape[1])
			roi = cv2.resize(roi, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
		#img = cv2.resize(roi, (150 , 50))
		img = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
		kernel = np.ones((1, 1), np.uint8)
		img = cv2.dilate(img, kernel, iterations=1)
		config = ("-l eng --oem 1 --psm 7")  #可在這切換成中文，但打印在框上變亂碼
		print(img.shape)
		text = pytesseract.image_to_string(img, config=config)
		resultstext.append(text)
		results.append(((startX, startY, endX, endY), text, i))
	results = sorted(results, key=lambda r:r[0][1])
	output = orig.copy()
	for ((startX, startY, endX, endY), text, i) in results:
		text = ''.join(text.split())
		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
		cv2.rectangle(output, (startX-6, startY-6), (endX+6, endY+6),(0, 0, 255), 2) #<-------------------------修改部分
		cv2.putText(output, str(i), (startX, startY - 20),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
		print(text)
	cv2.imshow('Image', output)
	key = cv2.waitKey(0)
	return(resultstext,output)
img = text_recognitioner(cv2.imread('C:/Users/Eric/Desktop/123/src/images/example_01.jpg'), cv2.dnn.readNet('C:/Users/Eric/Desktop/123/src/frozen_east_text_detection.pb'))


