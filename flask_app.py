from flask import Flask
from flask import request
from flask import jsonify
from prediccion import prediccion
import base64
import cv2
import array

app = Flask(__name__)

@app.route('/')
def index():
    return 'Reconocimiento de aves.'

@app.route('/predecir', methods = ['POST'])
def params():

    param = request.get_json()
    

    try:
        idImagen = param['idImagen']
    except:
        idImagen='imagen'

    try:
        imagen = param['imagen']
    except:
        return jsonify(status=False,error='Inserte una imagen')


    imageDecode=base64.b64decode(imagen)
    
    image_result = open('imagenDecodificada.jpg', 'wb') # create a writable image and write the decoding result
    image_result.write(imageDecode)

    reconocimiento=prediccion()

    imagenPrueba=cv2.imread("imagenDecodificada.jpg",1)

    
    try:
        imagenPrueba = cv2.cvtColor(imagenPrueba, cv2.COLOR_RGB2GRAY)
    except:
        return jsonify(status=False,error='Imagen no valida')
    
    
    categorias=["Gaviota occidental","Gallina","Gallinazo","Tacuarita","Mirlo de alas rojas","Green violetear","Cardenal norteño","Hapalopsittaca amazonina","Scarlet tanager","Barranquillo coronado","Mirlo de cabeza amarilla","Lazuli Bunting"]

    print("llego")
    indiceCategoria,predicciones = reconocimiento.predecir(imagenPrueba)
   
    return jsonify(status=True,idImagen=idImagen,prediccion=categorias[indiceCategoria],probabilidades=predicciones[0].tolist())

if __name__ == '__main__':
    app.run(debug = True)
