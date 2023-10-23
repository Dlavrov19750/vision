import cv2
import sys
import numpy as np

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Загрузите видеофайл, камера:
# свойства захвата камеры, при видиофайле игнорируются.
capture.set(cv2.CAP_PROP_FPS, 24)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# player = MediaPlayer(video_path)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_profile_cascade = cv2.CascadeClassifier(r'haarcascade_profileface.xml')

id = input("Введите свой номер:_")
# id = int(id)
count = 0
# Кадр - это изображение, которое вы хотите, флаг - это успех / неудача:


# Перебирайте кадры видео:


while True:
	flag, frame = capture.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))

	for (x, y, w, h) in faces:
		count += 1
		cv2.imwrite('dataset/User.' + str(id) + "." + str(count) + ".jpg", gray[y:y + h, x:x + w])
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
	# audio_frame, val = player.get_frame()

		if flag == 0:
			break

		# Escape to exit
	if count > 25:
		break

	cv2.waitKey(1) & 0xFF

	cv2.imshow("Front", frame)

capture.release()
cv2.destroyAllWindows()
#----------------------------------------------------------------------------------------------------------------------
count = 26
while True:
	flag, frame = capture.read()
	grayb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces_prof = face_profile_cascade.detectMultiScale(grayb, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
	for (x, y, w, h) in faces_prof:
		count += 1
		cv2.imwrite('dataset/User.' + str(id) + "." + str(count) + ".jpg", gray[y:y + h, x:x + w])
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
# audio_frame, val = player.get_frame()

		if flag == 0:
			break


# Escape to exit
	if count > 52:
		break

	cv2.waitKey(1) & 0xFF
	cv2.imshow("Profile", frame)

capture.release()
cv2.destroyAllWindows()

print("Все Данные профиля собраны")
# _______________________________________________________________________________________________________________________
