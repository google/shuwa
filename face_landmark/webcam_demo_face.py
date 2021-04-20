import sys; sys.path.insert(1, '../')
from common import crop_square

import time
import numpy as np
import cv2
from face_manager import FaceManager

WEBCAM_HEIGHT = 480
WEBCAM_WIDTH = 640

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 60)

    

face_manager = FaceManager()

    
while True:
    
    ret, frame = cap.read()
    frame = crop_square(frame)    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    t1 = time.time()
    face_flag, face_keypoints = face_manager(frame_rgb)
    t2 = time.time() - t1
    
    # use only 2D.
    face_keypoints = face_keypoints[:,:2]

    # draw.
    face_manager.draw_keypoints(frame, face_keypoints)
    
            
    cv2.putText(frame, "frame_time: {:.0f} ms".format(t2*1000), (10,50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("result", frame)
    

    key = cv2.waitKey(1)   
    if key == ord("q"):      
        cap.release()
        cv2.destroyAllWindows()
        break
    
