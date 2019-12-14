from tensorflow.python.keras.models import load_model
import numpy as np
import cv2

class prediccion():
    
    def __init__(self):
        self.rutaModelo="modeloPajaros.keras"
        self.model=load_model(self.rutaModelo)
        self.width=128
        self.heigth=128

    def predecir(self,imagen):
      
        imagen=cv2.resize(imagen,(self.width,self.heigth))
        imagen=imagen.flatten()
        imagen=np.array(imagen)
        imagenNormalizada=imagen/255
        pruebas=[]
        pruebas.append(imagenNormalizada)
        imagenesAPredecir=np.array(pruebas)
        predicciones=self.model.predict(x=imagenesAPredecir)
        claseMayorValor=np.argmax(predicciones,axis=1)
        return claseMayorValor[0],predicciones