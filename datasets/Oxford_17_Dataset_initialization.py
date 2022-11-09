
import numpy as np
import os, shutil

def retrieve_info(file, info_header, error_message): 
    try: 
        file_read = open(file, "r")
    except FileNotFoundError: 
        raise Exception(f"[Error] {file} does not exist. {error_message}")
        
    for line in file_read.readlines(): 
        if (len(line) > len(info_header)):
            if (line[0:len(info_header)] == info_header): 
                return(line[len(info_header) : len(line) : 1])
    file_read.close()
    return None

# Dataset Name
name = "Oxford_17_Original"

# Root path
root = retrieve_info("setup.txt", "Path: b'", "Make sure that you have finished the setup module in the beginning of the notebook. ")
root = root[0:len(root)-1:1]

# Dataset path
path = root + f"/datasets/{name}"

if (path == None): raise Exception(f"[Error] The root directory path does not exist. Make sure that you have finished the setup module in the beginning of the notebook. ")

if (os.path.exists(path)): shutil.rmtree(path, ignore_errors=True)

print ("Downloading Oxford 17 flowers dataset. ")
os.system("wget https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz")
os.system("tar zxvf 17flowers.tgz")
os.system(f"mv jpg datasets/{name}")
os.system("rm -rf 17flowers.tgz jpg")
print ("Successfully downloaded Oxford 17 flowers dataset. ")
print(f"Dataset path: {path}")

if (os.path.exists(path + "/train")): shutil.rmtree(path_1 + "/train", ignore_errors=True)
if (os.path.exists(path + "/val")): shutil.rmtree(path_1 + "/val", ignore_errors=True)
if (os.path.exists(path + "/test")): shutil.rmtree(path_1 + "/test", ignore_errors=True)

file_write = open(f"dataset_config.txt", "w")
file_write.write (f"Dataset: {name}")
file_write.close()

print ("Setting up dataset configurations. ")
import setup_dataset
print ("Dataset configurations are all set. ")


if (0):
    
    images = sorted([i for i in os.listdir(path) if "jpg" in i])
    print (f"There is a total of {len(images)} images in the Dataset.\n")
    images_perms = []
    for i in range (17): 
        images_perm = np.random.permutation(images[i*80: (i+1)*80])
        images_perms.append(images_perm)
    training_images_num = round(80 * training_image_ratio * dataset_ratio / (training_image_ratio + validation_image_ratio + testing_image_ratio))
    validation_images_num = round(80 * validation_image_ratio * dataset_ratio / (training_image_ratio + validation_image_ratio + testing_image_ratio))
    testing_images_num = int(80 * dataset_ratio) - training_images_num - validation_images_num

    
    try:
        os.makedirs(path + "/training", exist_ok = True)
        print ("The \"training\" folder: Successfully created. ")
        try:
            for i in range(17):
                class_name = "class_" + str(i + 1)
                os.makedirs(path + f"/training/{class_name}", exist_ok=True)
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



