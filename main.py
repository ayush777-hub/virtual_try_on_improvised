'''
This file implements the web server on flask.
It can provide web service, response all predict requests.
'''
import os
from flask import Flask, render_template, request, url_for, send_from_directory
from PIL import Image
import numpy as np
import time
import platform
from Model import Model
import random
import cv2
from io import StringIO, BytesIO
import base64
from datetime import datetime
import json
from flask_ngrok import run_with_ngrok
import matplotlib.image as mpimg
from camera import VideoCamera

# Use base64 to send & receive images between clients and the server
def readb64(base64_string):
    sbuf = BytesIO()
    sbuf.write(base64.b64decode(base64_string))
    res = Image.open(sbuf)
    return np.array(res)


def writeb64(img):
    img_str = cv2.imencode('.bmp', img)[1]
    imagebase64 = base64.b64encode(img_str)
    imagebase64 = bytes.decode(imagebase64)
    return imagebase64


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# init for all global variables

model = Model("checkpoints/jpp.pb",
              "checkpoints/gmm.pth",
              "checkpoints/tom.pth",
              use_cuda=False)

app = Flask(__name__)
run_with_ngrok(app)


# UPLOAD_FOLDER = 'request_upload'
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# cloth list for web server
cloth_list_raw = os.listdir(os.path.join(BASE_DIR, "static", "img"))
cloth_list = []
counter = 0
for cloth in cloth_list_raw:
    if 'jpeg' or 'jpg' in cloth:
        cloth_list.append([os.path.join("static", "img", cloth), counter])
        counter += 1

CART=[]


@app.route('/shirt')
def hello_world():
    return render_template('login.html', img_list=cloth_list)

@app.route('/accesories')
def index():
    return render_template('index.html')

@app.route('/checkOut')
def checkOut():
    return render_template('checkout.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route("/cart/<file_path>",methods = ['POST', 'GET'])
def cart(file_path):
    global CART
    file_path = file_path.replace(',','/')
    print("ADDED", file_path)
    CART.append(file_path)
    return render_template("checkout.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/tryon/<file_path>',methods = ['POST', 'GET'])
def tryon(file_path):
	file_path = file_path.replace(',','/')
	os.system('python tryOn.py ' + file_path)
	return render_template('checkout.html')
    #return redirect('http://127.0.0.1:5000/',code=302, Response=None)

@app.route('/tryall',methods = ['POST', 'GET'])
def tryall():
    print("YESSS")
    if request.method == 'POST':
        cart = request.form['mydata'].replace(',', '/')
        os.system('python test.py ' + cart)
        return render_template('checkout.html', message='')
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    print(request.form)
    print(request.files)
    if (not len(request.files) == 2 or (len(request.form) == 1 and len(request.files) == 1)):
        return render_template('login.html', info="selection error", img_list=cloth_list)
    else:
        index = 0  # init
        cloth_image = None
        if len(request.form) == 1:
            index = int(request.form['optionsRadios'][6:])
        person_image = request.files['person_image']
        if len(request.files) == 2:
            cloth_image = request.files['cloth_image']

        start_time = time.time()
        o_name, h_name = run_model_web(
            person_image, cloth_list[index][0].split("/")[-1], cloth_image)
        end_time = time.time()
        if o_name is None: # bad cloth image
            return 'I told you only clothes image with shape 256*192*3'
        else:
            return render_template('login.html', img_list=cloth_list, result1=h_name, result2=o_name, info="time: %.3f" % (end_time-start_time))


def run_model_web(f, cloth_name, cloth_f=None):
    '''
    prediction service. cloth_name and cloth_f cannot be both None. cloth_f is prior, which is from user upload.
    '''
    if cloth_f is None:
        print(f, cloth_name)
        c_img = mpimg.imread('/content/Virtual-Try-On-Flask/static/img/'+cloth_name)
    else:
        print(f, cloth_f)
        try:
            c_img = mpimg.imread(cloth_f)
        except:
            c_img = mpimg.imread('/content/Virtual-Try-On-Flask/static/img/'+cloth_name)

    # local resource temp file would be used as static resource.
    print(c_img.shape)
    if len(c_img.shape)!=3:
      return "THE IMAGE IS NOT SUPPORTED!!!"
    if c_img.shape[2]!=3:
      return "THE IMAGE IS NOT SUPPORTED!!!"
    c_img = cv2.resize(c_img,(192,256))
    temp_o_name = os.path.join("static","result","%d_%s" % (int(time.time()), cloth_name.split("/")[-1]))
    temp_h_name = os.path.join("static","human","%d_%s" % (int(time.time()), cloth_name.split("/")[-1]))

    if c_img.shape[0] != 256 or c_img.shape[1] != 192 or c_img.shape[2] != 3:
        return None, None

    img = mpimg.imread(f)
    img = cv2.resize(img,(192,256))
    human_img = img
    human_img = cv2.cvtColor(human_img,cv2.COLOR_RGB2BGR)
    c_img = cv2.cvtColor(c_img,cv2.COLOR_RGB2BGR)
    print(f"image shape:{human_img.shape}\n\n\n")
    out, v = model.predict(human_img, c_img, need_bright=False, keep_back=True)
    print("v:"+str(v))
    out = np.array(out,dtype=np.float32)

    path1 = '/content/Virtual-Try-On-Flask/'+temp_o_name
    path2 = '/content/Virtual-Try-On-Flask/'+temp_h_name
    if 'jpg' not in temp_o_name and 'jpeg' not in temp_o_name:
        path1 = path1 + '.jpeg'
        temp_o_name += '.jpeg'
    if 'jpg' not in temp_h_name and 'jpeg' not in temp_h_name:
        path2 = path2 + '.jpeg'
        temp_h_name += '.jpeg'
    cv2.imwrite(path1, out)
    cv2.imwrite(path2, human_img)
    return temp_o_name, temp_h_name


def getimg():
    data_str = request.data
    data_str = bytes.decode(data_str)
    data_str = data_str.replace('\n', '')
    data_json = json.loads(data_str)
    base64img_p = data_json['image_person']
    img_person = readb64(base64img_p)
    img_person = cv2.rotate(img_person, 2)
    img_person = cv2.flip(img_person, 1)
    base64img_c = data_json['image_cloth']
    img_cloth = readb64(base64img_c)
    return [img_person, img_cloth]


'''
json format example:
client:
    {
        'image_person':'...',
        'image_cloth':'...'
    }

server:
    {
        'status':'ok',
        'output_image':'...'
    }
'''
@app.route('/cloth', methods=['GET', 'POST'])
def Hello_cloth():
    '''
    响应客户端请求
    reponse requests from clients
    '''
    output_str = ""
    output_json = {}
    status = 'ok'
    if request.method == 'POST':
        # temp file would be writed to root dir
        input_person, input_cloth = getimg()
        cv2.imwrite('in.jpg', input_person)
        input_person = input_person[60:580, 45:435]
        cv2.imwrite('in_2.jpg', input_person)
        output_img, v = model.predict(input_person, input_cloth, need_bright=True, keep_back=True, need_dilate=True)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('out.jpg', output_img)
        print("v:"+str(v))
        output_base64 = writeb64(output_img)
        if v < 0.1: # confidence is too weak to show
            status = 'failure'
        else:
            status = 'ok'
        output_json["status"] = status
        output_json["output_image"] = output_base64
        output_str = json.dumps(output_json)
        return output_str
    return "please use http client to request!"


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True

    # run server locally
    app.run()

    # or as a servers
    # app.run(host='0.0.0.0', port=5000)