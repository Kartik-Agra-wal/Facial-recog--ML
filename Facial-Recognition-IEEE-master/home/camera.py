import cv2
import os
from django.conf import settings
from keras.models import load_model
import numpy as np
from PIL import Image
labels =  ["user1","unknown","kartik_agrawal"]

haar_cascade = cv2.CascadeClassifier('D:\\Facial recog -ML\\Facial-Recognition-IEEE-master\\home\\haar_face.xml')

class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
		self.model = load_model('D:\\Facial recog -ML\\save\\fine_tuning.h5')
		self.model.compile(loss='binary_crossentropy',
				optimizer='rmsprop',
				metrics=['accuracy'])
	def __del__(self):
		self.video.release()

	def get_frame(self):
		_, image = self.video.read()
		image1 = Image.fromarray(image,'RGB')
		image1.save('test.jpg')

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces_detected = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
		for (x, y, w, h) in faces_detected:
			cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
		frame_flip = cv2.flip(image,1)
		ret, jpeg = cv2.imencode('.jpg', frame_flip)


		img = cv2.imread('test.jpg')
		img = cv2.resize(img,(224,224))
		img = np.reshape(img,[1,224,224,3])
		predict_1 = self.model.predict(img)
		print(predict_1)
		#predict= predict_1[0]
		#class_labels=[labels[i] for i,prob in enumerate(predict) if prob > 0.5]
		#print(class_labels)
		os.remove("test.jpg")
		

		return jpeg.tobytes()
	
	#def recognition(self):
		#test_jpeg = self.get_frame()
		model = load_model('/save/fine_tunning.h5')
		model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
		img = cv2.imread(self.get_frame())
		img = cv2.resize(img,(320,240))
		img = np.reshape(img,[1,320,240,3])
		classes = model.predict_classes(img)
		print(classes)