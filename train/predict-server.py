import os
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms                                               

import io

from torch.utils.data import Dataset, DataLoader
from http.server import BaseHTTPRequestHandler, HTTPServer
from joblib import Parallel, delayed

from PIL import Image

import json, base64

from model import LinkNet

# ISIC image normalization constants                                                      
mean = [0.486, 0.336, 0.275]
std = [0.299, 0.234, 0.209]

val_transform = transforms.Compose([          
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
]) 

tensor2PIL = transforms.ToPILImage()

model_path = './models/resnet18_001_cpt_0.92907.pth'  # Path to the best model

def encode(array):
    buff = tensor2PIL(array)
    imgByteArr = io.BytesIO()
    buff.save(imgByteArr, format='JPEG')
    imgByteArr = imgByteArr.getvalue()
    return base64.b64encode(imgByteArr).decode('utf8')

device = torch.device("cuda") 
model = LinkNet(1, 3, 'resnet18_lite', 'sigmoid').to(device)
state = torch.load(model_path)
model.load_state_dict(state)
model.eval()

def predict(v_file):
    image = Image.open(v_file)
    image = val_transform(image)
    data = image.unsqueeze(0)    
    answer_key = {}
    with torch.no_grad():
        pred = model(data.to(device))
        tens = pred.detach().cpu()[0]
    res = encode(tens)
    answer_key = {"mask": res}
    return answer_key


#Create custom HTTPRequestHandler class
class FunHTTPRequestHandler(BaseHTTPRequestHandler):
  #handle GET command
  def do_GET(self):
    try:
        #send code 200 response
        self.send_response(200)
        #send header first
        self.send_header('Content-type','text-html')
        self.end_headers()
        print("Predicting file: ", self.path)
        answer_key = predict(self.path)
        self.wfile.write(bytes(json.dumps(answer_key), "utf8"))
        return

    except IOError:
      self.send_error(404, 'file not found')


def run():
  print('http server is starting at 127.0.0.1:8081...')
  server_address = ('127.0.0.1', 8081)
  httpd = HTTPServer(server_address, FunHTTPRequestHandler)
  print('http server is running...')
  httpd.serve_forever()

run()
