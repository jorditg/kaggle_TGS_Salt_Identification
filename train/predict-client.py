import sys
import json
from http.client import HTTPConnection

v_file = sys.argv[-1]
conn = HTTPConnection('127.0.0.1', 8081, timeout=1000)
conn.request('GET', v_file)
rsp = conn.getresponse()
data_received = json.loads(rsp.read().decode("utf-8"))

#print(json.dumps(data_received))
#with open('preds.json', 'w') as outfile:
    #json.dump(data_received, outfile)
conn.close()

import os
from PIL import Image
from io import BytesIO
import base64

im = Image.open(BytesIO(base64.b64decode(data_received['mask'])))
im.save(os.path.basename(v_file) + "_predicted_segmentation.jpeg", "JPEG")

