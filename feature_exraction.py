import cv2


def basic_features(contour):  # Calculate basic features like area, perimeter, and solidity
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return calculate_shape(area, perimeter, solidity)


hull = cv2.convexHull(contour)
hull_area = cv2.contourArea(hull)

solidity = area / hull_area if hull_area > 0 else 0



