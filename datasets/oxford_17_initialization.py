
# path 需要写中间文档存储 + 转化
# path_1 = path + "/Dataset"
# 我们应当支持 Dataset 自主命名

import numpy as np
import os, shutil


if (os.path.exists(path + "/Dataset")): 
    shutil.rmtree(path + "/Dataset", ignore_errors=True)

os.system("wget https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz")
os.system("tar zxvf 17flowers.tgz")
os.system("mv jpg Dataset")
os.system("rm -rf 17flowers.tgz")

path_1 = path + "/Dataset"
if (os.path.exists(path_1 + "/training")): shutil.rmtree(path_1 + "/training", ignore_errors=True)
if (os.path.exists(path_1 + "/validation")): shutil.rmtree(path_1 + "/validation", ignore_errors=True)
if (os.path.exists(path_1 + "/testing")): shutil.rmtree(path_1 + "/testing", ignore_errors=True)

images = [i for i in os.listdir(path_1) if "jpg" in i]
images = sorted(images)
print ("There is a total of", len(images), "images in the Dataset."); print ("")
images_perms = []
for i in range (17): 
    images_perm = np.random.permutation(images[i*80: (i+1)*80])
    images_perms.append(images_perm)
training_images_num = round(80 * training_image_ratio * dataset_ratio / (training_image_ratio + validation_image_ratio + testing_image_ratio))
validation_images_num = round(80 * validation_image_ratio * dataset_ratio / (training_image_ratio + validation_image_ratio + testing_image_ratio))
testing_images_num = int(80 * dataset_ratio) - training_images_num - validation_images_num
try:
    os.makedirs(path_1 + "/training", exist_ok = True)
    print ("The \"training\" folder: Successfully created. ")
    try:
        for i in range(17):
            class_name = "class_" + str(i + 1)
            os.makedirs(path_1 + f"/training/{class_name}", exist_ok=True)
        print ("The sub-folders under \"training\": Successfully created. ")
    except OSError: print ("Error in creating sub-folders under \"training\". ")      
except OSError: print ("Error in creating the \"training\" folder. ")
try:
    os.makedirs(path_1 + "/validation", exist_ok = True)
    print ("The \"validation\" folder: Successfully created. ")  
    try:
        for i in range(17):
            class_name = "class_" + str(i + 1)
            os.makedirs(path_1 + f"/validation/{class_name}", exist_ok=True)
        print ("The sub-folders under \"validation\": Successfully created. ")
    except OSError: print ("Error in creating sub-folders under \"validation\". ")
except OSError: print ("Error in creating the \"validation\" folder. ")
try:
    os.makedirs(path_1 + "/testing", exist_ok = True)
    print ("The \"testing\" folder: Successfully created. ")
    try:
        for i in range(17):
            class_name = "class_" + str(i + 1)
            os.makedirs(path_1 + f"/testing/{class_name}", exist_ok=True)
        print ("The sub-folders under \"testing\": Successfully created. ")
    except OSError: print ("Error in creating sub-folders under \"testing\". ")
except OSError: print ("Error in creating the \"testing\" folder. ")
training_images = []
validation_images = []
testing_images = []
for i in range (17):
    training_images.extend(images_perms[i][0:training_images_num])
    for j in range(int(training_images_num * dataset_ratio)):
        os.system(f"cp \"{path_1}/{images_perms[i][j]}\" \"{path_1}/training/class_{i+1}\"")
    validation_images.extend(images_perms[i][training_images_num : training_images_num + validation_images_num]) 
    for j in range(int(validation_images_num * dataset_ratio)):
        os.system(f"cp \"{path_1}/{images_perms[i][training_images_num + j]}\" \"{path_1}/validation/class_{i+1}\"")
    testing_images.extend(images_perms[i][training_images_num + validation_images_num : 80])
    for j in range(int(testing_images_num * dataset_ratio)):
        os.system(f"cp \"{path_1}/{images_perms[i][training_images_num + validation_images_num + j]}\" \"{path_1}/testing/class_{i+1}\"")
for i in range(len(images)): os.system(f"rm -rf \"{path_1}/{images[i]}\"")
os.system(f"rm -rf \"{path_1}/files.txt\"")
os.system(f"rm -rf \"{path_1}/files.txt~\"")