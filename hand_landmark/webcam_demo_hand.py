import sys; sys.path.insert(1, '../')
from common import crop_square

import time
import numpy as np
import cv2
from hand_manager import HandManager

WEBCAM_HEIGHT = 480
WEBCAM_WIDTH = 640

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 60)

hand_manager = HandManager()


while True:
    
    ret, frame = cap.read()
    frame = crop_square(frame)    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    t1 = time.time()
    hand_flag, hand_keypoints = hand_manager([frame_rgb])
    t2 = time.time() - t1
        
    # use only one hand.
    hand_keypoints = hand_keypoints[0]

    # use only 2D.
    hand_keypoints = hand_keypoints[:,:2]
        
    # Draw.
    hand_manager.draw_keypoints(frame, hand_keypoints)
    
            
    cv2.putText(frame, "frame_time: {:.0f} ms".format(t2*1000), (10,50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
   
    cv2.imshow("result", frame)
    

    key = cv2.waitKey(1)   
    if key == ord("q"):      
        cap.release()
        cv2.destroyAllWindows()
        break
    
