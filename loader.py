import json


def load_annotations(file_path): # Load COCO annotations from a JSON file
    with open(file_path, 'r') as f: # Open the specified JSON file for reading
        data = json.load(f)
    return data


def get_image_info(data, image_id): # Retrieve image information from the COCO dataset based on the image ID
    for img in data["images"]:
        if img["id"] == image_id:
            return img
    return None
