import cv2
import numpy as np
from PIL import Image
import os
from imutils import paths

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataset'
def getimageID(puth):
	imagePath = [os.path.join(path, f) for f in os.listdir(puth)]
	faces = []
	ids = []
	for imagePath in imagePath:
		faceImage = Image.open(imagePath).convert('L')
		faceNP = np.array(faceImage)
		Id = (os.path.split(imagePath)[-1].split(".")[1])
		Id = int(Id)
		faces.append(faceNP)
		ids.append(Id)
		print(faceNP)
		cv2.imshow("Train", faceNP)
		cv2.waitKey(1)
	return  ids ,faces

IDs,facedata = getimageID(path)
recognizer.train(facedata, np.array(IDs))
recognizer.write('Trainer.yml')

cv2.destroyAllWindows()