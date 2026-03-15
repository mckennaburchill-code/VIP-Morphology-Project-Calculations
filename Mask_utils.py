import numpy as np
import cv2


def polygon_to_mask(segmentation, height, width): # Convert polygon segmentation to a binary mask
    mask = np.zeros((height, width), dtype=np.uint8) #Reshape the flat list of coordinates in a 2D array
    points = np.array(segmentation).reshape(-1, 2).astype(np.int32) #Fill the polygon defined by the points in the mask
    cv2.fillPoly(mask, [points], 255) #Return the binary mask at 255 for the polygon area and 0 elsewhere
    return mask 

def get_contour(mask): #Extract the contour from the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0] if contours else None   #Return the first contour found, or None if no contours are detected 
