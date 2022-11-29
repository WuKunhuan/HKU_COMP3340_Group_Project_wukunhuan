
import os
import torchvision
from torchvision import transforms
from IPython.display import clear_output

class import_dataset: 
    
    def __init__(self, setup, dataset): 
        
        self.root = setup.path
        self.dataset = dataset

        all_dataset = [f.name for f in os.scandir(f"{self.root}/datasets") if f.is_dir() and f.name != "__pycache__" and f.name != ".ipynb_checkpoints"]
        all_dataset = sorted(all_dataset)
        print (f"Current datasets: {all_dataset}")
        print (f"Import dataset {self.dataset} ... ", end="")
        if (not self.dataset in all_dataset): 
            raise Exception (f"[Error] {self.dataset} not found in the datasets folder. ")
        self.path = f"{self.root}/datasets/{self.dataset}"
        
        self.path = f"{self.root}/datasets/{self.dataset}"
        self.train_path = self.path + f"/train"
        self.val_path = self.path + f"/val"
        self.test_path = self.path + f"/test"

        file_read = open(f"{self.root}/configurations/dataset_train_transform.txt", "r"); train_transform = ""; 
        for line in file_read.readlines(): 
            train_transform += line
        file_read.close(); 

        file_read = open(f"{self.root}/configurations/dataset_val_transform.txt", "r"); val_transform = ""; 
        for line in file_read.readlines(): 
            val_transform += line
        file_read.close(); 

        file_read = open(f"{self.root}/configurations/dataset_test_transform.txt", "r"); test_transform = ""; 
        for line in file_read.readlines(): 
            test_transform += line
        file_read.close(); 
                
        self.train_dataset = torchvision.datasets.ImageFolder(root=self.train_path, transform=eval(train_transform))
        self.val_dataset = torchvision.datasets.ImageFolder(root=self.val_path, transform=eval(val_transform))
        self.test_dataset = torchvision.datasets.ImageFolder(root=self.test_path, transform=eval(test_transform))
        
        print (f"Finished\nA total of {len(self.train_dataset)} train, {len(self.val_dataset)} val and {len(self.test_dataset)} test images. ")

        

