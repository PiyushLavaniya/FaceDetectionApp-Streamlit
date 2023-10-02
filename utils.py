import tensorflow as tf
import numpy as np 
import cv2
from tensorflow.keras.models import load_model



def detection_video(model, video):

    #video = cv2.VideoCapture(video)
    while video.isOpened():
            ret , frame = video.read()
            frame = frame[50:500, 50:500,:]
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = tf.image.resize(rgb, (120,120))
                
            yhat = model.predict(np.expand_dims(resized/255,0))
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

            
            frame = cv2.resize(frame, (0, 0), fx = 0.8, fy = 0.8)

    return frame
        #cv2.imshow('facetracker', frame)
        
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
            
    #cap.release()
    #cv2.destroyAllWindows()

def detections_image(model, frame):

    frame = frame[50:500, 50:500,:]
            
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))
            
    yhat = model.predict(np.expand_dims(resized/255,0))
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
            
    return frame

#model = load_model('facedetection.h5')
#detection(model)