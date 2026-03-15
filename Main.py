from loader import load_annotations
from Mask_utils import polygon_to_mask, get_contour
from feature_extraction import basic_features
from cust_detection import radical_cusp_detection, rank_cusps


def main():
    data = load_annotations("annotations.json")

    annotation = data["annotations"][0] # Assuming we want to process the first annotation for demonstration
    image_id = data["images"][annotation["image_id"]]

    segmentation = annotation["segmentation"][0]
    height = image_id["height"]
    width = image_id["width"] # of iterations: 1

    mask = polygon_to_mask(segmentation, height, width)
    contour = get_contour(mask)
    peaks, radical_dist, centroid = radical_cusp_detection(contour)
    ranked_peaks = rank_cusps(peaks, radical_dist)
    features = basic_features(contour) # of iterations: 1

    print("Detected Cusps:", ranked_peaks)
    print("Features", features)

if __name__ == "__main__":
    main()
    print("Program executed successfully")  # Ensuring the program executed without errors
