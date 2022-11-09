
import numpy as np
import os, shutil

class Oxford_17_Dataset: 
    
            
    def retrieve_info(self, file, info_header, error_message): 
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
    
    def __init__(self, type): 
        self.name = ""
        self.path = ""
        self.type = type
        
        if (self.type == "Original"): 
            
            self.root = self.retrieve_info ("setup.txt", "Path: b'", "")
            self.root = self.root[0:len(self.root)-1:1]
            self.name = "Oxford_17_Original"
            self.path = self.root + f"/datasets/{self.name}"
            
        elif (self.type == "Copies"):
            
            self.root = self.retrieve_info("setup.txt", "Path: b'", "")
            self.root = self.root[0:len(self.root)-1:1]

            new_name = ""

            os.system("clear")
            while (new_name == ""): 
                
                n = input("Input a name for the copy of Oxford 17 flowers dataset. ")
                for i in range(len(n)): 
                    if ((n[i] >= '0' and n[i] <= '9') or (n[i] >= 'a' and n[i] <= 'z') or (n[i] >= 'A' and n[i] <= 'Z') or (n[i] == '_')):
                        new_name += n[i]
                if (new_name == ""): continue
                new_name = "Oxford_17_" + new_name
                all_dataset = [f.name for f in os.scandir(f"{self.root}/datasets") if f.is_dir()]
                # print(all_dataset)
                if (new_name in all_dataset): 
                    input (f"The dataset {new_name} already exists. ")
                    new_name = ""
                confirm = ""
                confirm = input(f"Name: {new_name}. Press Enter to confirm. Enter anything to discard. ")
                if (confirm != ""): new_name = ""
                    
            new_path = self.root + f"/datasets/{new_name}"
            print (f"The new dataset {new_name} is created. ")
            print (f"Dataset path: {new_path}")
            self.name = new_name
            self.path = new_path
        
        else: raise Exception (f"[Error] Invalid type: {self.type}")

        if (os.path.exists(self.path)): shutil.rmtree(self.path, ignore_errors=True)
        print ("Downloading Oxford 17 flowers source data. ")
        os.system("wget https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz")
        os.system("tar zxvf 17flowers.tgz")
        os.system(f"mv jpg datasets/{self.name}")
        os.system("rm -rf 17flowers.tgz")
        print ("Successfully downloaded Oxford 17 flowers source data. ")

        print(f"Dataset path: {self.path}")
        if (os.path.exists(self.path + "/train")): shutil.rmtree(self.path + "/train", ignore_errors=True)
        if (os.path.exists(self.path + "/val")): shutil.rmtree(self.path + "/val", ignore_errors=True)
        if (os.path.exists(self.path + "/test")): shutil.rmtree(self.path + "/test", ignore_errors=True)

        self.oxford_17_configure()

        print ("Generating Oxford 17 flowers dataset. ")

        images = sorted([i for i in os.listdir(self.path) if "jpg" in i])
        print (f"There is a total of {len(images)} images in the Oxford 17 flowers source data.\n")
        images_perms = []
        for i in range (17): 
            images_perm = np.random.permutation(images[i*80: (i+1)*80])
            images_perms.append(images_perm)
        file_read = open("setup_dataset.txt", "r")
        num_classes = 0
        train, val, test = 0, 0, 0
        dataset_ratio = 1
        for line in file_read.readlines():
            if (len(line) > 16 and line[0:16] == "Num of classes: "): 
                num_classes = self.retrieve_info("setup_dataset.txt", "Num of classes: ", "")
                num_classes = int(num_classes)
                continue
            if (len(line) > 16 and line[0:16] == "Train_Val_Test: "): 
                train, val, test = self.retrieve_info("setup_dataset.txt", "Train_Val_Test: ", "").split(" ")
                train = int(train); val = int(val); test = int(test)
                continue
            if (len(line) > 15 and line[0:15] == "Dataset ratio: "): 
                dataset_ratio = self.retrieve_info("setup_dataset.txt", "Dataset ratio: ", "")
                dataset_ratio = float(dataset_ratio)
                continue
        train_num = round(80 * train * dataset_ratio / (train + val + test))
        val_num = round(80 * val * dataset_ratio / (train + val + test))
        test_num = int(80 * dataset_ratio) - train_num - val_num

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


    def oxford_17_configure(self): 
        print ("Setting up dataset configurations. ")
        import setup_dataset as S
        S.setup_dataset(self.name)
        print ("Dataset configurations are all set. ")


