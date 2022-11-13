
import os
from IPython.display import clear_output

class setup_dataset: 

    def __init__(self, dataset): 
        
        input ("(enter anything to start setting dataset configurations)")

        Oxford = False
        if (len(dataset) > 10): 
            if (dataset[0:10] == "Oxford_17_"):
                num_classes = 17
                Oxford = True
        if (not Oxford): 
            clear_output(wait=True)
            print("Input an positive integer to indicate the total number of classes in the dataset. ")
            num_classes = None
            while (num_classes == None):
                clear_output(wait=True)
                num_classes = input("Number of classes in the dataset: ")
                if (not num_classes.isdigit()): 
                    num_classes = None
                else: num_classes = int(num_classes)
        self.num_classes = num_classes
        
        
        train, val, test, dataset_ratio = None, None, None, None
        self.train, self.val, self.test, self.dataset_ratio = None, None, None, None
        
        # When the dataset is the Oxford original, we need to distribute images into train, val and test. 
        # When the dataset is not the Oxford original (i.e., DIY), no need to distribute images. 
        # Instead, users have to place images into defined folders by themselves. 
        
        if (Oxford):
            
            while ((train and val and test) == None):
                clear_output(wait=True)
                print ("Give 3 positive integers, seperated by a space, to indicate the ratio of training, validation and testing images")
                print ("For example, input \"8 1 1\" will set the ratio 8:1:1. ")
                try: 
                    train, val, test = input("{training : validation : testing} = ").split(' ')
                except: train, val, test = None, None, None
                if (train != None): 
                    if (not train.isdigit()): 
                        train = None
                    else: train = int(train)
                if (val != None):
                    if (not val.isdigit()): 
                        val = None
                    else: val = int(val)
                if (test != None):
                    if (not test.isdigit()): 
                        test = None
                    else: test = int(test)
            self.train = train
            self.val = val
            self.test = test

            while (dataset_ratio == None): 
                clear_output(wait=True)
                print ("Give a ratio in range (0, 1] to indicate the proportion of the dataset to be used. ")
                print ("For example, input \"0.6\" will use 60% of the dataset. ")                
                try:
                    dataset_ratio = float(input("Dataset ratio: "))
                    if (not (0 < dataset_ratio and dataset_ratio <= 1)):
                        dataset_ratio = None
                except ValueError: dataset_ratio = None
            self.dataset_ratio = dataset_ratio
            input ("(enter anything to complete the configuration)")
        
        else: 
            print (f"train, val and test dataset folders of your DIY Dataset has been created. ")
            print (f"You still need to put images for each class under the train, val and test datasets")
            input ("(enter anything to complete the configuration)")

    
    def return_setup(self):
        return (self.num_classes, self.train, self.val, self.test, self.dataset_ratio)

