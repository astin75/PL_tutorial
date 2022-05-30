import glob
import random
import json
from tqdm import tqdm
fld_name = "natural_images"
class_folder = glob.glob(fld_name+"/*")
test_set_ratio = 0.2
train_set = {}
test_set = {}
test_set['dataset'] = []
train_set['dataset'] = []
one_hot = {}




# split image from class
for index_i, class_fld in tqdm(enumerate(class_folder)):
    class_images = glob.glob(class_fld+"/*.jpg")
    random.shuffle(class_images)
    test_size = len(class_images) * test_set_ratio
    one_hot[str(index_i)] = class_fld.replace("\\","/").split("/")[1]
    # split train/test
    for index_j, j in enumerate(class_images):
        value = {"class": class_fld.replace("\\","/").split("/")[1],
                 "class_num": index_i,
                 "path": j.replace("\\","/")}
        if index_j <= test_size:
            test_set["dataset"].append(value)
        else:
            train_set["dataset"].append(value)
test_set['one_hot'] = one_hot
train_set['one_hot'] = one_hot
#save images..
with open("train.json", "w") as f:
    json.dump(train_set,f, indent=4)
with open("test.json", "w") as f:
    json.dump(test_set,f, indent=4)