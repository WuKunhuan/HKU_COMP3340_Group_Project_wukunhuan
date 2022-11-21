
import numpy as np
import os, shutil
from IPython.display import clear_output

class DIY_Dataset: 
    
    def __init__(self, setup): 
        
        file_read = open(setup.path +"/configurations/dataset.txt")
        all_dataset = [f.name for f in os.scandir(f"{setup.path}/datasets" ) if f.is_dir()  and f.name != '__pycache__' and f.name != '.ipynb_checkpoints']
        print (f"Current datasets: {all_dataset}")
        
        # Read the dataset name, train_val_test and dataset_ratio
        for line in file_read.readlines():
            if (len(line) > 6 and line[0:6] == "[name]"): self.name = line[6:len(line)-1]
            if (len(line) > 16 and line[0:16] == "[train_val_test]"): self.train, self.val, self.test = line[16:len(line)-1].split(' ')
            if (len(line) > 15 and line[0:15] == "[dataset_ratio]"): self.dataset_ratio = float(line[15:len(line)-1])
        file_read.close(); 
        
        self.train, self.val, self.test = None, None, None
        self.path = setup.path + f"/datasets/{self.name}"
        print (f"New dataset name: {self.name}")
        print (f"New dataset path: {self.path}")
        if (self.name in all_dataset): 
            print ("")
            print ("The dataset name already exists. Are you sure to overwrite the dataset? ")
            i = input("Press enter to conform or enter anything to quit. ")
            if (i != ""): raise Exception ("User quit the dataset setup. ")
        

        print ("\n")
        if (os.path.exists(self.path + "/train")): shutil.rmtree(self.path + "/train", ignore_errors=True)
        if (os.path.exists(self.path + "/val")): shutil.rmtree(self.path + "/val", ignore_errors=True)
        if (os.path.exists(self.path + "/test")): shutil.rmtree(self.path + "/test", ignore_errors=True)
    
        images = sorted([i for i in os.listdir(self.path) if "jpg" in i])
        print (f"Generating train, val and test datasets... ", end = "")
        
        images_perms = []
        for i in range (17): 
            images_perm = np.random.permutation(images[i*80: (i+1)*80])
            images_perms.append(images_perm)
            
        os.makedirs(self.path + "/train", exist_ok = True)
        os.makedirs(self.path + "/val", exist_ok = True)
        os.makedirs(self.path + "/test", exist_ok = True)
        
        for i in range(17):
            class_name = "class_" + str(i + 1)
            os.makedirs(self.path + f"/train/{class_name}", exist_ok=True)
            os.makedirs(self.path + f"/val/{class_name}", exist_ok=True)
            os.makedirs(self.path + f"/test/{class_name}", exist_ok=True)
            
        print ("Finished")
        print (f"\nSuccessfully set up the {self.name} DIY dataset. You still need to put your images under the train, val and test directories. ")
        


