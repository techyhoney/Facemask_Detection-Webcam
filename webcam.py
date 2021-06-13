import cv2
import imghdr
import os
import numpy as np 
import streamlit as st
from imutils import paths
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image,ImageEnhance
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, ClientSettings

# -------------------------- SIDE BAR --------------------------------
SIDEBAR_OPTION_WEBCAM = "Webcam Capture"

# SIDEBAR_OPTIONS = [SIDEBAR_OPTION_WEBCAM]

st.sidebar.image("logo.png")
st.sidebar.write(" ------ ")

# --------------------------- Functions ----------------------------------

class VideoTransformer(VideoTransformerBase):
    def detect_and_predict_mask(self,frame1,net,model):
        #grab the dimensions of the frame and then construct a blob
        (h,w)=frame1.shape[:2]
        blob=cv2.dnn.blobFromImage(frame1,1.0,(300,300),(104.0,177.0,123.0))

        net.setInput(blob)
        detections=net.forward()

        #initialize our list of faces, their corresponding locations and list of predictions

        faces=[]
        locs=[]
        preds=[]


        for i in range(0,detections.shape[2]):
            confidence=detections[0,0,i,2]


            if confidence>0.7:
            #we need the X,Y coordinates
                box=detections[0,0,i,3:7]*np.array([w,h,w,h])
                (startX,startY,endX,endY)=box.astype('int')

                #ensure the bounding boxes fall within the dimensions of the frame
                (startX,startY)=(max(0,startX),max(0,startY))
                (endX,endY)=(min(w-1,endX), min(h-1,endY))

                #extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it
                face=frame1[startY:endY, startX:endX]
                face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
                face=cv2.resize(face,(96,96))
                face=img_to_array(face)
                face=preprocess_input(face)

                faces.append(face)
                locs.append((startX,startY,endX,endY))

            #only make a predictions if atleast one face was detected
        if len(faces)>=1:
            faces=np.array(faces,dtype='float32')
            preds=model.predict(faces,batch_size=12)

        return (locs,preds)
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        prototxtPath = 'deploy.prototxt.txt'
        weightsPath= 'res10_300x300_ssd_iter_140000.caffemodel'
        net=cv2.dnn.readNet(weightsPath,prototxtPath)
        model=load_model(r'model/custom_4000_32_100.h5')
        (locs,preds)=self.detect_and_predict_mask(img,net,model)
        for (box,pred) in zip(locs,preds):
            (startX,startY,endX,endY)=box
            (mask,withoutMask)=pred
            label='Mask' if mask>withoutMask else 'No Mask'
            color=(0,255,0) if label=='Mask' else (0,0,255)
            #include the probability in the label
            label="{}: {:.2f}%".format(label,max(mask,withoutMask)*100)
            img=cv2.putText(img,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
            img=cv2.rectangle(img,(startX,startY),(endX,endY),color,2)
        return img




# ------------------------- Selection From SideBar ------------------

st.sidebar.markdown('## Webcam Capture')

app_mode = SIDEBAR_OPTION_WEBCAM

if app_mode == SIDEBAR_OPTION_WEBCAM:
    ctx=webrtc_streamer(client_settings=ClientSettings(
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
                ),
                key="face-mask-detection", 
                video_transformer_factory=VideoTransformer
               )

