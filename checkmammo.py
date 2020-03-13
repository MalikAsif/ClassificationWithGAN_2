# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:24:35 2020

@author: Asrock
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:45:45 2020

@author: Asif
"""
import tensorflow.keras
from tensorflow.keras.models import load_model
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit, QFrame
import sys 
from PyQt5.QtGui import QPixmap
import numpy as np
from keras.preprocessing import image 
import cv2

 
class Window(QWidget):
    def __init__(self):
        super().__init__()
 
        self.title = "Breast Cancer Classification App"
        self.top = 200
        self.left = 500
        self.width = 400
        self.height = 300
        self.InitWindow()
        self.np_image = np.empty([128, 128])
 
 
    def InitWindow(self):
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.new_model = tensorflow.keras.models.load_model('D:/Breast Cancer/asif/Code/trained_models/model.h5')
        vbox = QVBoxLayout()
 
        self.btn1 = QPushButton("Upload Mammogram")
        self.btn1.clicked.connect(self.getImage)
        self.btn2 = QPushButton("Check Result")
        self.btn2.clicked.connect(self.getResult)
        vbox.addWidget(self.btn1)
        vbox.addWidget(self.btn2)
        
        self.label = QLabel("mammogram")
        vbox.addWidget(self.label)
        self.setLayout(vbox)
        self.show()
 
    def getImage(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file','E:\\' , "Image files (*.jpg *.png)")
        imagePath = fname[0]
        pixmap = QPixmap(imagePath)
        #img = image.load_img(imagePath, target_size = (224, 224))
        img = image.load_img(imagePath, target_size = (128, 128), grayscale= 'True')
        img = image.img_to_array(img)
        a = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.np_image = np.array(a).astype('float32')/255
        self.np_image = np.expand_dims(self.np_image, axis=0)
        self.np_image = np.expand_dims(self.np_image, axis=-1)
        pixmap1 = pixmap.scaled(400, 200)
        self.label.setPixmap(QPixmap(pixmap1))
        #self.resize(pixmap.width(), pixmap.height())
        
    def getResult(self):
        #pixmap = QPixmap(imagePath)
        #self.label.setPixmap(QPixmap(pixmap))
        r = self.modelResult()
        for i in r:
            c =i
            break
        print(c[0])
        if (c[0] > c[1]):
            pixmap = QPixmap('D:/Breast Cancer/asif/Breast-cancer-detection-in-mammograms-master/Breast-cancer-detection-in-mammograms-master/Img/cancer.png')
        else:
            pixmap = QPixmap('D:/Breast Cancer/asif/Breast-cancer-detection-in-mammograms-master/Breast-cancer-detection-in-mammograms-master/Img/normal.png')
        self.label.setPixmap(QPixmap(pixmap))
        self.resize(pixmap.width(), pixmap.height())
        self.show()  
    def modelResult(self):
        prediction = self.new_model.predict(self.np_image)
        print(prediction[0])
        return prediction


App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())