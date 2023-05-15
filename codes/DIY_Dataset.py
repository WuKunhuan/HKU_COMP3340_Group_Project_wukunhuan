
import numpy as np
import os, shutil
from IPython.display import clear_output

class DIY_Dataset: 
    
    def __init__(self, setup, name): 
        
        all_dataset = [f.name for f in os.scandir(f"{setup.path}/datasets" ) if f.is_dir()  and f.name != '__pycache__' and f.name != '.ipynb_checkpoints']
        print (f"Current datasets: {all_dataset}")
        
        self.name = name
        self.path = setup.path + f"/datasets/{self.name}"
        print (f"New dataset name: {self.name}")
        print (f"New dataset path: {self.path}")
        if (self.name in all_dataset): 
            print ("")
            print ("The dataset name already exists. Are you sure to overwrite the dataset? ")
            i = input("Press enter to conform or enter anything to quit. ")
            if (i != ""): 
                print ("User quit the dataset setup. ")
                return; 
        

        print ("\n")
        if (os.path.exists(self.path + "/train")): shutil.rmtree(self.path + "/train", ignore_errors=True)
        if (os.path.exists(self.path + "/val")): shutil.rmtree(self.path + "/val", ignore_errors=True)
        if (os.path.exists(self.path + "/test")): shutil.rmtree(self.path + "/test", ignore_errors=True)
    
        print (f"Generating train, val and test datasets ... ", end = "")
            
        os.makedirs(self.path + "/train", exist_ok = True)
        os.makedirs(self.path + "/val", exist_ok = True)
        os.makedirs(self.path + "/test", exist_ok = True)
        
        for i in range(17):
            class_name = "class_" + str(i + 1)
            os.makedirs(self.path + f"/train/{class_name}", exist_ok=True)
            os.makedirs(self.path + f"/val/{class_name}", exist_ok=True)
            os.makedirs(self.path + f"/test/{class_name}", exist_ok=True)
            
        print ("Finished")
        print (f"\nSuccessfully set up the {self.name} DIY dataset. \nYou still need to put your images under the train, val and test directories. ")
        


