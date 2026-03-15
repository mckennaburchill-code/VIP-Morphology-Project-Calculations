from loader import load_annotations, get_image_info
from Mask_utils import polygon_to_mask, get_contour
from feature_extraction import basic_features
from cust_detection import radical_cusp_detection, rank_cusps

def main():
    data = load_annotations("annotations.json")

    # container for whatever you want to collect
    all_features = []
    all_ranks    = []

    for ann in data["annotations"]:
        img = get_image_info(data, ann["image_id"])
        if img is None:
            continue          # defensive, should not happen in a well‑formed COCO file

        seg = ann["segmentation"][0]          # flat [x0,y0,x1,y1,…]
        h, w = img["height"], img["width"]

        mask    = polygon_to_mask(seg, h, w)
        contour = get_contour(mask)

        peaks, rad_dist, centroid = radical_cusp_detection(contour)
        ranked_peaks              = rank_cusps(peaks, rad_dist)
        feats                     = basic_features(contour)

        all_ranks.append(ranked_peaks)
        all_features.append(feats)

        print(f"Image {img['file_name']} – cusps:", ranked_peaks)
        print("    features:", feats)
        save_path = f"results/{img['file_name']}_features.txt"

    # do something with the collected results, e.g. save to disk
    # ...  

if __name__ == "__main__":
    main()
    print("Program executed successfully")  # Ensuring the program executed without errors
