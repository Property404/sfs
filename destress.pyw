#/usr/bin/env python
import cv2
import sys
import logging as log
import datetime as dt
import alt_facializer as facializer
import mom

# Basic setup
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='geo.log',level=log.INFO)

log.info("Starting video capture")
video_capture = cv2.VideoCapture(1)
anterior = 0
stress = 0
stress_threshold = 40
message_wait = 0
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

		if anterior != len(faces):
			anterior = len(faces)
		#cv2.imwrite('bla.jpg', frame[y:y+h, x:x+w])
                cv2.imwrite("bla.jpg", frame)
                state = facializer.getState("bla.jpg")[0]
                if state == "stressed":
                    stress+=1
                    if message_wait > 0:
                        message_wait -= 1
                else:
                    stress = 0;
                sys.stdout.write(str(stress)+state+" \r")
                if stress >= stress_threshold:
                    if message_wait <= 0:
                        mom.chill()
                        stress = 0
                        message_wait = 40

	# Display the resulting frame
	#cv2.imshow('Video', frame)


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	# Display the resulting frame
	#cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
