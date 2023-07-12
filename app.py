import cv2  
from matplotlib import pyplot as plt
import pytesseract
from PIL import Image
from flask import Flask, request, render_template, redirect,jsonify
import urllib.request

app = Flask(__name__)
from werkzeug.utils import secure_filename

@app.route('/', methods=["GET", "POST"])

def upload_image():

    global img
    try:
        reqRef = request.get_json(force=True)
        img = reqRef["imgUrl"]
        urllib.request.urlretrieve(img,"gfg.jpeg")
    # img = Image.open("gfg.jpeg")
    # img.show()

    # print("abc"+ img)
    
        
        
        path_to_tesseract=r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        #Imagepath='gfg.jpeg'
        pytesseract.tesseract_cmd=path_to_tesseract
        text=pytesseract.image_to_string(Image.open("gfg.jpeg"))

        return jsonify({
            "ocr_result": text,

        })

    except Exception as e:
        return jsonify({
            "weigthURL": "No found",
            "error": str(e)
        })







if __name__ == '__main__':
    app.run(debug=False, port='0.0.0.0')