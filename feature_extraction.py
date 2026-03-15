import cv2


def basic_features(contour):  # Calculate basic features like area, perimeter, and solidity
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    solidity = area / hull_area if hull_area > 0 else 0

    return {
        "area": area,
        "perimeter": perimeter,
        "solidity": solidity
    }
