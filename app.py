import tensorflow as tf
import numpy as np
import streamlit as st
import cv2
from PIL import Image
import tempfile
from tensorflow.keras.models import load_model
from utils import detection_video, detections_image


DEMO_IMAGE = 'person_face.jpg'
DEMO_VIDEO = 'WIN_20230803_11_31_17_Pro.mp4'


def main():
    
    st.set_page_config(page_title='face_detection', page_icon=':boy:', layout='wide')
    st.header("Object Detection")


    with st.sidebar:
        st.header('You can select from one of the options below : ')
    

    app_mode = st.sidebar.selectbox('Choose the Detection Object',
                                    ['About Application', 'Face Detector'])
    
    st.sidebar.info('Select one you want to run it on from the Dropdown Menu below.')

    if app_mode == 'About Application':
        st.subheader('You can select one of the options from the Sidebar, based upon what do you want to Detect.')
        st.markdown("This application is made using **Tensorflow** as base and **Streamlit** is used to provide the required GUI.")


    if app_mode == 'Face Detector':
        st.subheader('This app is designed to detect only your face. But if you want to detect any other object, you can use the source code provided in github. You can use this algorithm to detect anything you want.')
        st.markdown('Source Code is provided in the **Github** repo.')
        facetracker = load_model('facedetection.h5')
        st.sidebar.markdown('----')

        st.markdown('**Detected Faces**')

        file_mode = st.sidebar.selectbox('Select the type of file',
                             ['Run on Image', 'Run on Video'])
        
        if file_mode == 'Run on Image':
            img = st.sidebar.file_uploader('Upload an image', type=["jpg", "jpeg", "png"])

            if img is not None:
                image = np.array(Image.open((img)))

            else:
                demo_image = DEMO_IMAGE
                image = np.array(Image.open((demo_image)))

            st.sidebar.image(image)

            detections_image(facetracker, image)

            st.image(image)

            st.info('If it is not detecting the face then try a picture with a slight lower resolution.')
            st.info('This model is trained on images with resolution of (640X480). So please keep a note of that while testing out this model.')


        elif file_mode == 'Run on Video':
            use_webcam = st.sidebar.button('Use webcam')
            record = st.sidebar.checkbox('Record Video')

            if record:
                st.checkbox('Recording', value = True)

            #setting up an empty frame to fill it later
            stframe = st.empty()

            video_file_buffer = st.sidebar.file_uploader('Upload a Video', type = ["mp4", "wav", "m4v", "asf", "avi"])

            #That name can be retrieved from the name member of the file object. 
            # Means that you can get the name of the temporary file created

            temp_file = tempfile.NamedTemporaryFile(delete = False)  

            #to get our input Video 
            if not video_file_buffer:
                if use_webcam:
                    video = cv2.VideoCapture(0)

                else:
                    video = cv2.VideoCapture(DEMO_VIDEO)

                    temp_file.name = DEMO_VIDEO 

            else:
                temp_file.write(video_file_buffer.read())
                video = cv2.VideoCapture(temp_file.name)


            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_input = int(video.get(cv2.CAP_PROP_FPS))

            #now recording a video codec is a hardware or software that compresses and decompresses digital video, 
            #to make file sizes smaller and storage and distribution of the videos easier.
            codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')

            output_video_buffer = cv2.VideoWriter('output1.mp4', codec, fps_input, (width,height))

            st.sidebar.text('Input Video')
            st.sidebar.video(temp_file.name)

            while video.isOpened():
                ret , frame = video.read()
                frame = frame[50:500, 50:500,:]
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = tf.image.resize(rgb, (120,120))
                
                yhat = facetracker.predict(np.expand_dims(resized/255,0))
                sample_coords = yhat[1][0]
                
                if yhat[0] > 0.5: 
                    # Controls the main rectangle
                    cv2.rectangle(frame, 
                                tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                                tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
                                        (255,0,0), 2)
                    # Controls the label rectangle
                    cv2.rectangle(frame, 
                                tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
                                                [0,-30])),
                                tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                                [80,0])), 
                                        (255,0,0), -1)
                    
                    # Controls the text rendered
                    cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                                        [0,-5])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)



                if record:
                    output_video_buffer.write(frame)
            
                frame = cv2.resize(frame, (0, 0), fx = 0.8, fy = 0.8)
                stframe.image(frame, channels = "BGR", use_column_width=True)
        



if __name__=='__main__':
    main()