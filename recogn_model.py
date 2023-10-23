import cv2
import sys
import numpy as np



capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Загрузите видеофайл, камера:
# свойства захвата камеры, при видиофайле игнорируются.
capture.set(cv2.CAP_PROP_FPS, 24)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600)


# player = MediaPlayer(video_path)
face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
face_fac_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

name_list = ["", "Dmitriy", "Тааанюрик", "Софья"]

while True:
	flag, frame = capture.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
	faces = face_fac_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
	for (x, y, w, h) in faces:
		serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
		if conf > 50:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
			#сv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
			cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
			cv2.putText(frame, name_list[serial], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
			print(name_list[serial])
		else:

			# cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
			# #cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
			# cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
			cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
# audio_frame, val = player.get_frame()
	if flag == 0:
		break
	cv2.waitKey(1) & 0xFF
	cv2.imshow("Video", frame)

capture.release()
cv2.destroyAllWindows()
