import cv2  
from matplotlib import pyplot as plt
import pytesseract
from PIL import Image
from flask import Flask, request, render_template, redirect,jsonify
import numpy as np
import urllib.request
#from firebase_admin import credentials, firestore, initialize_app

app = Flask(__name__)
#cred = credentials.Certificate('key.json')
#default_app = initialize_app(cred)
#db = firestore.client()
#todo_ref = db.collection('todos')

#app.config["IMAGE_UPLOADS"] = "C:/Users/Sathvik/Downloads/OCR/images"
# app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG","JPG","JPEG"]

from werkzeug.utils import secure_filename


#image_file = "images/vc2.png"
#img = cv2.imread(image_file)
@app.route('/', methods=["GET", "POST"])


def upload_image():

    global img
    reqRef = request.get_json(force=True)
    img = reqRef["imgUrl"]
    urllib.request.urlretrieve(img,"gfg.png")
    # img = Image.open("gfg.png")
    # img.show()

    # print("abc"+ img)
    try:
        
        
        inverted = 0
        img = cv2.imread("gfg.png")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("gray.jpg", gray)

        thresh, im_bw = cv2.threshold(img, 210, 230, cv2.THRESH_BINARY)
        cv2.imwrite("bw_image.jpg", im_bw)

        # def noise_removal(image):
        kernel = np.ones((1,1), np.uint8)
        im_bw = cv2.dilate(im_bw, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        im_bw = cv2.erode(im_bw, kernel, iterations=1)
        im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, kernel)
        im_bw = cv2.medianBlur(im_bw, 3)
            # return (image)
        
        no_noise = im_bw
        cv2.imwrite("no_noise.jpg", no_noise)

        img_file = "vc2.png"
        no_noise = "no_noise.jpg"

        img = Image.open(no_noise)
        ocr_result = pytesseract.image_to_string(img)

        #print(ocr_result)

        return jsonify({
            "ocr_result": ocr_result,

        })

    except Exception as e:
        return jsonify({
            "weigthURL": "No found",
            "error": str(e)
        })







if __name__ == '__main__':
    app.run(debug=False, port=2000)
