from turtle import distance # IGNORE

import numpy as np
from scipy.signal import find_peaks

def radical_cusp_detection(contour, min_distance=20):  
    # Calculate the curvature of the contour
   pts = contour.squeeze() #squeeze the contour to get a 2D array of points
   centroid = pts.mean(axis=0) #Calculate the centroid of the contour points

   radical_dist = np.linalg.norm(pts - centroid, axis=1) #Calculate the distance from each contour point to the centroid
   peaks, properties = find_peaks(radical_dist, distance=min_distance)  

   return peaks, radical_dist[peaks], centroid

def rank_cusps(peaks, distances):
    # Sort peaks by distances in descending order
    sorted_indices = np.argsort(distances)[::-1]
    return peaks[sorted_indices], distances[sorted_indices]
