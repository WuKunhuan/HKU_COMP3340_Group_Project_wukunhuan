
# This file should be imported in the root directory. 

import os

class setup_dataset: 

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

        print("")
        print ("Give 3 positive integers, seperated by a space, to indicate the ratio of training, validation and testing images")
        print ("For example, input \"8 1 1\" will set the ratio 8:1:1. ")
        train, val, test = None, None, None
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

        print("")
        print ("Give a ratio in range (0, 1] to indicate the proportion of the dataset to be used. ")
        print ("For example, input \"0.6\" will use 60% of the dataset. ")
        dataset_ratio = None
        while (dataset_ratio == None): 
            try:
                dataset_ratio = float(input("Dataset ratio: "))
                if (not (0 < dataset_ratio and dataset_ratio <= 1)):
                    dataset_ratio = None
            except ValueError: dataset_ratio = None

        file_write = open(f"setup_dataset.txt", "w")
        file_write.write(f"Num of classes: {num_classes}\n")
        file_write.write(f"Train_Val_Test: {train} {val} {test}\n")
        file_write.write(f"Dataset ratio: {dataset_ratio}\n")
        file_write.close()

        print("")

