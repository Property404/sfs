#/usr/bin/env python
import cv2
import sys
import logging as log
import datetime as dt
import serial

# Basic setup
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='geo.log',level=log.INFO)

log.info("Starting video capture")
video_capture = cv2.VideoCapture(0)
anterior = 0
while True:
	if not video_capture.isOpened():
		log.info('Unable to load camera.')
		pass

    # Capture frame-by-frame
	ret, frame = video_capture.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
	)

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		center_x = (2*x+w)/2
		center_y = (2*y+h)/2
		width = w
		height = h
		#cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		#cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), 2)
		print([center_x, center_y])
		if center_x > 350:
			print("Please turn camera right");
		elif center_x < 250:
			print("Please turn camera left");

		if anterior != len(faces):
			anterior = len(faces)
		print(frame)
		cv2.imwrite('bla.jpg', frame[y:y+h, x:x+w])

	# Display the resulting frame
	cv2.imshow('Video', frame)


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	# Display the resulting frame
	cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
