
import os
import torchvision
from torchvision import transforms
from IPython.display import clear_output

class import_dataset: 
    
    def __init__(self, setup): 
        
        self.root = setup.path
        all_dataset = [f.name for f in os.scandir(f"{self.root}/datasets") if f.is_dir() and f.name != "__pycache__" and f.name != ".ipynb_checkpoints"]
        all_dataset = sorted(all_dataset)
        
        if (len(all_dataset) == 0): 
            self.dataset = None
            self.path = None
            self.train_path = None
            self.train_dataset = None
            self.val_path = None
            self.val_dataset = None
            self.test_path = None
            self.test_dataset = None
            clear_output (wait=True)
            print ("You have not created any dataset! Please create at least 1 dataset(s) to continue. ")
            print ("Use the above \"Setup dataset\" module to create datasets. ")
        else: 
            dataset_select = None
            while (dataset_select == None):
                clear_output(wait=True)
                print ("Select one dataset to import by entering a number below. ")
                for j in range (len(all_dataset)):
                    print (f"Enter {j}: {all_dataset[j]}")
                try: 
                    dataset_select = int(input("\nYour choice: "))
                    if (not (0 <= dataset_select and dataset_select < len(all_dataset))): 
                        dataset_select = None
                except ValueError: dataset_select = None
            self.dataset = all_dataset[dataset_select]
            self.path = f"{self.root}/datasets/{self.dataset}"
            self.train_path = self.path + f"/train"
            self.val_path = self.path + f"/val"
            self.test_path = self.path + f"/test"

            flag = False; 
            while (not flag): 
                clear_output(wait=True) 
                input (f"Change dataset image transformation in Pytorch in: \n\"{self.root}/resources/dataset_train_transform.txt\"\n\"{self.root}/resources/dataset_val_transform.txt\"\n\"{self.root}/resources/dataset_test_transform.txt\". \n(After you finished, enter anything to continue)")

                file_read = open(f"{self.root}/resources/dataset_train_transform.txt", "r"); train_transform = ""; 
                for line in file_read.readlines(): 
                    train_transform += line
                file_read.close(); 

                file_read = open(f"{self.root}/resources/dataset_val_transform.txt", "r"); val_transform = ""; 
                for line in file_read.readlines(): 
                    val_transform += line
                file_read.close(); 

                file_read = open(f"{self.root}/resources/dataset_test_transform.txt", "r"); test_transform = ""; 
                for line in file_read.readlines(): 
                    test_transform += line
                file_read.close(); 
                
                clear_output(wait=True)
                print(f"Transform to the train dataset: {train_transform}\n")
                print(f"Transform to the val dataset: {val_transform}\n")
                print(f"Transform to the test dataset: {test_transform}\n")
                print("Press Enter to confirm or enter anything to discard. ")
                flag = input()
                if (flag == ""): 
                    flag = True
                    self.train_dataset = torchvision.datasets.ImageFolder(root=self.train_path, transform=eval(train_transform))
                    self.val_dataset = torchvision.datasets.ImageFolder(root=self.val_path, transform=eval(val_transform))
                    self.test_dataset = torchvision.datasets.ImageFolder(root=self.test_path, transform=eval(test_transform))
                else:
                    flag = False
            clear_output(wait=True)
            print ("Import datasets successfully. ")
            print ("There is a total of " + str(len(self.train_dataset)) + " training images. ")   
            print ("There is a total of " + str(len(self.val_dataset)) + " validation images. ")
            print ("There is a total of " + str(len(self.test_dataset)) + " testing images. ")

        

