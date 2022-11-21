
import os, shutil
import numpy as np

class setup_default_dataset:
    
    def __init__(self, setup): 
        
        print ("Creating the Oxford_17 dataset ... ", end = "")
        
        # Setup the Oxford_17 dataset
        train, val, test = 8, 1, 1
        
        self.path = setup.path + f"/datasets/Oxford_17"
    
        os.system("wget https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz")
        os.system("tar zxvf 17flowers.tgz")
        os.system("mv jpg datasets/Oxford_17")
        os.system("rm -rf 17flowers.tgz")
        
        
        if (os.path.exists(self.path + "/train")): shutil.rmtree(self.path + "/train", ignore_errors=True)
        if (os.path.exists(self.path + "/val")): shutil.rmtree(self.path + "/val", ignore_errors=True)
        if (os.path.exists(self.path + "/test")): shutil.rmtree(self.path + "/test", ignore_errors=True)
            
        images = sorted([i for i in os.listdir(self.path) if "jpg" in i])
        images_perms = []
        for i in range (17): 
            images_perm = np.random.permutation(images[i*80: (i+1)*80])
            images_perms.append(images_perm)
        train_num = round(80 * train * 1 / (train + val + test))
        val_num = round(80 * val * 1 / (train + val + test))
        test_num = int(80 * 1) - train_num - val_num
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
            train_images.extend(images_perms[i][0: train_num])
            for j in range(int(train_num * 1)):
                os.system(f"cp \"{self.path}/{images_perms[i][j]}\" \"{self.path}/train/class_{i+1}\"")
            val_images.extend(images_perms[i][train_num : train_num + val_num]) 
            for j in range(int(val_num * 1)):
                os.system(f"cp \"{self.path}/{images_perms[i][train_num + j]}\" \"{self.path}/val/class_{i+1}\"")
            test_images.extend(images_perms[i][train_num + val_num : 80])
            for j in range(int(test_num * 1)):
                os.system(f"cp \"{self.path}/{images_perms[i][train_num + val_num + j]}\" \"{self.path}/test/class_{i+1}\"")
                
        for i in range(len(images)): os.system(f"rm -rf \"{self.path}/{images[i]}\"")
        os.system(f"rm -rf \"{self.path}/files.txt\"")
        os.system(f"rm -rf \"{setup.path}/datasets/jpg\"")
        
        print ("Finished")
        
        print ("\nCOMP3340_GP default datasets are all created. ")