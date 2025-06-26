# read label.png and dumo the mask of obj
import os
from PIL import Image

ob_id_to_names = {
    1: "ape",
    2: "benchvise",
    3: "bowl",
    4: "camera",
    5: "can",
    6: "cat",
    7: "cup",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}

names_to_ob_id = {v: k for k, v in ob_id_to_names.items()}

root = "data/lm/real_train_sample_64"

objs = os.listdir(root)

for obj in objs:
    obj_id = names_to_ob_id[obj]
    mask_dir = os.path.join(root, obj)
    label_files = os.listdir(mask_dir)
    label_files = [f for f in label_files if f.endswith("-label.png")]
    # label -> mask
    # set label == obj_id to 255, else 0
    for label_file in label_files:
        # make mask
        make_file = label_file.replace("-label.png", "-mask.png")
        label_path = os.path.join(mask_dir, label_file)
        mask_path = os.path.join(mask_dir, make_file)
        label = Image.open(label_path)
        mask = Image.new("L", label.size)
        for i in range(label.size[0]):
            for j in range(label.size[1]):
                if label.getpixel((i, j)) == obj_id:
                    mask.putpixel((i, j), 255)
                else:
                    mask.putpixel((i, j), 0)
        mask.save(mask_path)
