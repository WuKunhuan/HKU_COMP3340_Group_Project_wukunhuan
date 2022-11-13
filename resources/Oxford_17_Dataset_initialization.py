
import numpy as np
import os, shutil
from IPython.display import clear_output

class Oxford_17_Dataset: 
    
    def __init__(self, setup, type): 
        
        self.name = ""
        self.path = ""
        self.type = type
        
        if (self.type == "Original"): 
            
            self.root = setup.path
            self.name = "Oxford_17_Original"
            self.path = self.root + f"/datasets/{self.name}"
            
        elif (self.type == "Copies"):
            
            clear_output(wait=True)
            self.root = setup.path

            new_name = ""
            while (new_name == ""): 
                
                clear_output(wait=True)
                all_dataset = [f.name for f in os.scandir(f"{self.root}/datasets" ) if f.is_dir()  and f.name != '__pycache__' and f.name != '.ipynb_checkpoints']
                print (f"Current datasets: {all_dataset}")
                
                n = input("Input a name for the copy of Oxford 17 flowers dataset besides \"Original\". ")
                for i in range(len(n)): 
                    if ((n[i] >= '0' and n[i] <= '9') or (n[i] >= 'a' and n[i] <= 'z') or (n[i] >= 'A' and n[i] <= 'Z') or (n[i] == '_')):
                        new_name += n[i]
                if (new_name == ""): continue
                new_name = "Oxford_17_" + new_name
                all_dataset = [f.name for f in os.scandir(f"{self.root}/datasets" ) if f.is_dir()]
                
                if (new_name in all_dataset): 
                    print (f"The dataset {new_name} already exists. ")
                    new_name = ""
                    input ("(enter anything to give another name)")
                else: 
                    confirm = ""
                    print (f"Preferred name of the copy: {new_name}. ")
                    confirm = input(f"Press Enter to confirm. Enter anything to discard.")
                    if (confirm != ""): new_name = ""
                        
            new_path = self.root + f"/datasets/{new_name}"
            self.name = new_name
            self.path = new_path
            print (f"The copy {self.name} will be created under {self.path}. ")
            input ("(enter anything to start generating the copy)")

        
        else: raise Exception (f"[Error] Invalid type: {self.type}")

        clear_output(wait=True)
        if (os.path.exists(self.path)): shutil.rmtree(self.path, ignore_errors=True)
        print ("Downloading Oxford 17 flowers source data ...")
        os.system("wget https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz")
        os.system("tar zxvf 17flowers.tgz")
        os.system(f"mv jpg datasets/{self.name}")
        os.system("rm -rf 17flowers.tgz")
        print ("Successfully downloaded Oxford 17 flowers source data. ")
        if (os.path.exists(self.path + "/train")): shutil.rmtree(self.path + "/train", ignore_errors=True)
        if (os.path.exists(self.path + "/val")): shutil.rmtree(self.path + "/val", ignore_errors=True)
        if (os.path.exists(self.path + "/test")): shutil.rmtree(self.path + "/test", ignore_errors=True)
        
        num_classes, train, val, test, dataset_ratio = self.oxford_17_configure()
        
        clear_output(wait=True)
        # print ("Generating Oxford 17 flowers dataset ...")
        images = sorted([i for i in os.listdir(self.path) if "jpg" in i])
        print (f"There is a total of {len(images)} images in the Oxford 17 flowers source data.")
        print (f"We will use {round(dataset_ratio * 100)}% of them. ")
        print (f"Distributing images into train, val and test datasets ({train}:{val}:{test})...")
        
        images_perms = []
        for i in range (num_classes): 
            images_perm = np.random.permutation(images[i*80: (i+1)*80])
            images_perms.append(images_perm)
        train_num = round(80 * train * dataset_ratio / (train + val + test))
        val_num = round(80 * val * dataset_ratio / (train + val + test))
        test_num = int(80 * dataset_ratio) - train_num - val_num
        os.makedirs(self.path + "/train", exist_ok = True)
        os.makedirs(self.path + "/val", exist_ok = True)
        os.makedirs(self.path + "/test", exist_ok = True)
        for i in range(num_classes):
            class_name = "class_" + str(i + 1)
            os.makedirs(self.path + f"/train/{class_name}", exist_ok=True)
            os.makedirs(self.path + f"/val/{class_name}", exist_ok=True)
            os.makedirs(self.path + f"/test/{class_name}", exist_ok=True)
        train_images, val_images, test_images = [], [], []
        for i in range (num_classes):
            train_images.extend(images_perms[i][0:train_num])
            for j in range(int(train_num * dataset_ratio)):
                os.system(f"cp \"{self.path}/{images_perms[i][j]}\" \"{self.path}/train/class_{i+1}\"")
            val_images.extend(images_perms[i][train_num : train_num + val_num]) 
            for j in range(int(val_num * dataset_ratio)):
                os.system(f"cp \"{self.path}/{images_perms[i][train_num + j]}\" \"{self.path}/val/class_{i+1}\"")
            test_images.extend(images_perms[i][train_num + val_num : 80])
            for j in range(int(test_num * dataset_ratio)):
                os.system(f"cp \"{self.path}/{images_perms[i][train_num + val_num + j]}\" \"{self.path}/test/class_{i+1}\"")
        for i in range(len(images)): os.system(f"rm -rf \"{self.path}/{images[i]}\"")
        os.system(f"rm -rf \"{self.path}/files.txt\"")
        os.system(f"rm -rf \"{self.root}/datasets/jpg\"")
        
        clear_output(wait=True)
        print (f"Successfully generated the {self.name} dataset. ")
        

    def oxford_17_configure(self): 
        clear_output(wait=True)
        import resources.setup_dataset as S
        setup = S.setup_dataset(self.name)
        return setup.return_setup()


