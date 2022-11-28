
import numpy as np
import os, shutil
from IPython.display import clear_output

class Oxford_17_Dataset: 
    
    def __init__(self, setup, name, train, val, test, ratio): 
        
        all_dataset = [f.name for f in os.scandir(f"{setup.path}/datasets" ) if f.is_dir()  and f.name != '__pycache__' and f.name != '.ipynb_checkpoints']
        
        self.name = name
        self.train = train
        self.val = val
        self.test = test
        self.dataset_ratio = ratio
        
        self.train, self.val, self.test = float(self.train), float(self.val), float(self.test)
        self.path = setup.path + f"/datasets/{self.name}"
        print (f"New dataset name: {self.name}")
        print (f"New dataset train_val_test: {self.train} {self.val} {self.test}")
        print (f"New dataset dataset_ratio: {self.dataset_ratio}")
        print (f"New dataset path: {self.path}")
        if (self.name in all_dataset): 
            print ("")
            print ("The dataset name already exists. Are you sure to overwrite the dataset? ")
            i = input("Press enter to conform or enter anything to quit. ")
            if (i != ""): 
                print ("User quit the dataset setup. ")
                return; 
        

        print ("\n")
        if (os.path.exists(self.path)): shutil.rmtree(self.path, ignore_errors=True)
        print ("Downloading Oxford 17 flowers source data ... ", end = "")
        os.system("wget https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz > /dev/null")
        os.system("tar zxvf 17flowers.tgz")
        os.system(f"mv jpg datasets/{self.name}")
        os.system("rm -rf 17flowers.tgz")
        
        print ("Finished")
        if (os.path.exists(self.path + "/train")): shutil.rmtree(self.path + "/train", ignore_errors=True)
        if (os.path.exists(self.path + "/val")): shutil.rmtree(self.path + "/val", ignore_errors=True)
        if (os.path.exists(self.path + "/test")): shutil.rmtree(self.path + "/test", ignore_errors=True)
    
        images = sorted([i for i in os.listdir(self.path) if "jpg" in i])
        print (f"Distributing images into train, val and test datasets ({self.train}:{self.val}:{self.test})... ", end = "")
        
        images_perms = []
        for i in range (17): 
            images_perm = np.random.permutation(images[i*80: (i+1)*80])
            images_perms.append(images_perm)
            
        self.train_num = round(80 * self.train * self.dataset_ratio / (self.train + self.val + self.test))
        self.val_num = round(80 * self.val * self.dataset_ratio / (self.train + self.val + self.test))
        self.test_num = int(80 * self.dataset_ratio) - self.train_num - self.val_num
        
        
        os.makedirs(self.path + "/train", exist_ok = True)
        os.makedirs(self.path + "/val", exist_ok = True)
        os.makedirs(self.path + "/test", exist_ok = True)
        
        for i in range(17):
            class_name = "class_" + str(i + 1)
            os.makedirs(self.path + f"/train/{class_name}", exist_ok=True)
            os.makedirs(self.path + f"/val/{class_name}", exist_ok=True)
            os.makedirs(self.path + f"/test/{class_name}", exist_ok=True)
            
        train_images, val_images, test_images = [], [], []
        for i in range (17):
            train_images.extend(images_perms[i][0:self.train_num])
            for j in range(int(self.train_num * self.dataset_ratio)):
                os.system(f"cp \"{self.path}/{images_perms[i][j]}\" \"{self.path}/train/class_{i+1}\"")
            val_images.extend(images_perms[i][self.train_num : self.train_num + self.val_num]) 
            for j in range(int(self.val_num * self.dataset_ratio)):
                os.system(f"cp \"{self.path}/{images_perms[i][self.train_num + j]}\" \"{self.path}/val/class_{i+1}\"")
            test_images.extend(images_perms[i][self.train_num + self.val_num : int(80 * self.dataset_ratio)])
            for j in range(int(self.test_num * self.dataset_ratio)):
                os.system(f"cp \"{self.path}/{images_perms[i][self.train_num + self.val_num + j]}\" \"{self.path}/test/class_{i+1}\"")
                
        for i in range(len(images)): os.system(f"rm -rf \"{self.path}/{images[i]}\"")
        os.system(f"rm -rf \"{self.path}/files.txt\"")
        os.system(f"rm -rf \"{setup.path}/datasets/jpg\"")
        
        f = open(f"{self.path}/dataset_description.txt", "w")
        f.write(f"[Dataset Info]\n")
        f.write(f"Dataset_name: {self.dataset}\n")
        f.write(f"train_val_test: {self.train} {self.val} {self.test}\n")
        
        
        print ("Finished")
        print (f"\nSuccessfully set up the {self.name} dataset. ")
        


