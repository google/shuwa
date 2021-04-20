
import cv2
import numpy as np
def crop_square(image, ):        
    height, width, _ = image.shape
    
    if height < width:    
        start_x = width//2 - height//2
        end_x =  width//2 + height//2
        image = image[:, start_x:end_x]
     

    elif width < height:   
        start_y = height-width
        image = image[start_y:, :]       
        
    
    return image


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.shape[0:2][::-1]
    w, h = size, size 
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_CUBIC)
    new_image = np.zeros((size, size, 3), np.uint8)
    new_image.fill(128)
    dx = (w-nw)//2
    dy = (h-nh)//2
    new_image[dy:dy+nh, dx:dx+nw,:] = image
    return new_image