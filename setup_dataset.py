
# This file should be imported in the root directory. 

import os

class setup_dataset: 

    def __init__(self, dataset): 

        flag = False
        if (len(dataset) > 10): 
            if (dataset[0:10] == "Oxford_17_"):
                num_classes = 17
                flag = True
        if (not flag): 
            print("")
            print("Input an positive integer to indicate the total number of classes in the dataset. ")
            num_classes = None
            while (num_classes == None):
                os.system("clear")
                num_classes = input("Number of classes in the dataset: ")
                if (not num_classes.isdigit()): 
                    num_classes = None
                else: num_classes = int(num_classes)
        self.num_classes = num_classes
        
        train, val, test, dataset_ratio = None, None, None, None
        self.train, self.val, self.test, self.dataset_ratio = None, None, None, None
        if (flag):
            print("")
            print ("Give 3 positive integers, seperated by a space, to indicate the ratio of training, validation and testing images")
            print ("For example, input \"8 1 1\" will set the ratio 8:1:1. ")
            while ((train and val and test) == None):
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

            print("")
            print ("Give a ratio in range (0, 1] to indicate the proportion of the dataset to be used. ")
            print ("For example, input \"0.6\" will use 60% of the dataset. ")

            while (dataset_ratio == None): 
                try:
                    dataset_ratio = float(input("Dataset ratio: "))
                    if (not (0 < dataset_ratio and dataset_ratio <= 1)):
                        dataset_ratio = None
                except ValueError: dataset_ratio = None
            self.dataset_ratio = dataset_ratio

    
    def return_setup(self):
        return (self.num_classes, self.train, self.val, self.test, self.dataset_ratio)

