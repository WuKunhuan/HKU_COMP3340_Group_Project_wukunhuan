
import os
from IPython.display import clear_output
import math

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

class train_model:
    
    def __init__ (self, setup): 
        
        self.root = setup.path
        clear_output(wait=True)
        
        all_model = [f.name for f in os.scandir(f"{self.root}/models") if f.is_file() and len(f.name) > 3 and f.name[len(f.name)-3:len(f.name)] == ".py"]
        for i in range (len(all_model)): 
            all_model[i] = all_model[i][0:len(all_model[i]) - 3]
        all_model = sorted(all_model)
        
        if (len(all_model) == 0): 
            
            self.model = None
            self.model_loss_function = None
            self.model_optimizer = None
            self.train_dataloader = None
            self.val_dataloader = None
            self.test_dataloader = None
            clear_output (wait=True)
            print ("You have not created any model! Please create at least 1 model(s) to continue. ")
            print ("Your model should be in .py file, containing a class definition as introduced in the notebook. ")

        else: 
            
            finish = False
            
            while (not finish): 
                
                clear_output(wait=True)
                model_select = None

                while (model_select == None):
                    clear_output(wait=True)
                    print ("Select one model to train by entering a number below. ")
                    for j in range (len(all_model)):
                        print (f"Enter {j}: {all_model[j]}")
                    try: 
                        print("")
                        model_select = int(input("Your choice: "))
                        if (not (0 <= model_select and model_select < len(all_model))): 
                            model_select = None
                    except ValueError: model_select = None
                self.model_name = all_model[model_select]
                exec(f"import models.{self.model_name} as {self.model_name}")
                self.model = eval(f"{self.model_name}.{self.model_name} (num_classes = 17)")
                if (self.model == None):
                    raise Exception ("[Error] Generating the model failed. ")

                
                # https://neptune.ai/blog/pytorch-loss-functions
                all_loss_function = ["CrossEntropyLoss"]
                model_loss_select = None
                while (model_loss_select == None):
                    clear_output(wait=True)
                    print ("Select one model loss function by entering a number below. ")
                    for j in range (len(all_loss_function)):
                        print (f"Enter {j}: {all_loss_function[j]}")
                    try: 
                        print("")
                        model_loss_select = int(input("Your choice: "))
                        if (not (0 <= model_loss_select and model_loss_select < len(all_loss_function))): 
                            model_loss_select = None
                    except ValueError: model_loss_select = None
                self.model_loss_function = eval(f"nn.{all_loss_function[model_loss_select]}()")


                all_model_optimizer = ["Adam", "SGD"]
                model_optimizer_select = None
                while (model_optimizer_select == None):
                    clear_output(wait=True)
                    print ("Select one model optimizer by entering a number below. ")
                    for j in range (len(all_model_optimizer)):
                        print (f"Enter {j}: {all_model_optimizer[j]}")
                    try: 
                        print("")
                        model_optimizer_select = int(input("Your choice: "))
                        if (not (0 <= model_optimizer_select and model_optimizer_select < len(all_model_optimizer))): 
                            model_optimizer_select = None
                    except ValueError: model_optimizer_select = None
                self.model_optimizer_name = all_model_optimizer[model_optimizer_select]

                
                model_optimizer_lr = None
                while (model_optimizer_lr == None):
                    clear_output(wait=True)
                    print (f"Set {self.model_optimizer_name} optimizer's learning rate. ")
                    try: 
                        print("")
                        model_optimizer_lr = float(input("Learning rate: "))
                        if (not (0 < model_optimizer_lr)): 
                            model_optimizer_lr = None
                    except ValueError: model_optimizer_lr = None
                self.model_optimizer_lr = model_optimizer_lr
                
                
                if (self.model_optimizer_name == "SGD"): 
                    model_optimizer_momentum = None
                    while (model_optimizer_momentum == None):
                        clear_output(wait=True)
                        print (f"Set {self.model_optimizer_name} optimizer's momentum in range [0, 1). ")
                        try: 
                            time.sleep(100)
                            model_optimizer_momentum = float(input("\nMomentum: "))
                            if (not (0 <= model_optimizer_momentum and model_optimizer_momentum < 1)): 
                                model_optimizer_momentum = None
                        except ValueError: model_optimizer_momentum = None
                    self.model_optimizer_momentum = model_optimizer_momentum
                else: self.model_optimizer_momentum = None

                import torch.optim as optim
                if (self.model_optimizer_name == "SGD"): 
                    self.model_optimizer = eval(f"optim.{self.model_optimizer_name}(self.model.parameters(), lr=self.model_optimizer_lr, momentum={self.model_optimizer_momentum})")
                else: self.model_optimizer = eval(f"optim.{self.model_optimizer_name}(self.model.parameters(), lr=self.model_optimizer_lr)")
                
                model_epoch = None
                while (model_epoch == None):
                    clear_output(wait=True)
                    print (f"Set the number of epoches of training the {self.model_name} model (postive integer). ")
                    try: 
                        model_epoch = int(input("\nEpoches: "))
                        if (not (0 < model_epoch)): 
                            model_epoch = None
                    except ValueError: model_epoch = None
                self.train_model_epoch = model_epoch

                model_batch_size = None
                while (model_batch_size == None):
                    clear_output(wait=True)
                    print (f"Set the batch size of your {self.model_name} model's datasets (postive integer). ")
                    try: 
                        model_batch_size = int(input("\nBatch size: "))
                        if (not (0 < model_batch_size)): 
                            model_batch_size = None
                    except ValueError: model_batch_size = None
                self.train_model_batch_size = model_batch_size
                
                clear_output(wait=True)
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                if (self.device == "cpu"): print("[Device] Warning: Train the model with CPU because cuda is not available")
                else: print("[Device] Train the model with GPU")
                self.model.to(self.device)
                
                clear_output(wait=True)
                print ("All model training settings are completed. Please confirm: ")
                print (f"Model name: {self.model_name} (output classes = 17)")
                print (f"Loss function: {self.model_loss_function}")
                print (f"Optimizer: {self.model_optimizer_name} (learning rate={self.model_optimizer_lr}, momentum={self.model_optimizer_momentum})")
                print ("\nFor the model training: ")
                print (f"Epoches: {self.train_model_epoch}")
                print (f"Batch size: {self.train_model_batch_size}")
                if torch.cuda.is_available(): print (f"Device: GPU")
                else: print (f"Device: CPU")
                
                print()
                
                choice = input ("Press Enter to confirm or enter anything to discard. ")
                if (choice == ""): finish = True
    
    
    def print_train_val_plot (self, option, epoch, batch): 
        
        clear_output(wait=True)
        
        if (option == "train"): 
            print (f"Training the model {self.model_name} (batch_size = {self.train_model_batch_size})")
        if (option == "val"): 
            print (f"Validating the model {self.model_name} (batch_size = {self.train_model_batch_size})")
        if (option == "finish"): 
            print (f"Finished training {self.model_name} (batch_size = {self.train_model_batch_size})")
        
        progress_bar_length = 30; 
        progress_char = '#'
        remaining_char = '-'
        
        print ("[Epoch] ", end = "")
        progress = epoch * 1.0 / self.train_model_epoch * progress_bar_length
        
        for i in range (0,  math.floor(progress), 1): 
            print (progress_char, end = "")
        for i in range (0,  progress_bar_length - math.floor(progress), 1): 
            if (i == 0):
                print (math.floor(10 * (progress - math.floor(progress))), end="")
            else: 
                print (remaining_char, end = "")
        print (f" {epoch} out of {self.train_model_epoch}")
        
        if (option == "train"): 
            print ("[Batch] ", end = "")
            progress = batch * 1.0 / len(self.train_dataloader) * progress_bar_length
            
            for i in range (0,  math.floor(progress), 1): 
                print (progress_char, end = "")
            for i in range (0,  progress_bar_length - math.floor(progress), 1): 
                if (i == 0):
                    print (math.floor(10 * (progress - math.floor(progress))), end="")
                else: 
                    print (remaining_char, end = "")
                
            print (f" {batch} out of {len(self.train_dataloader)}")
        if (option == "val"): 
            print ("[Batch] ", end = "")
            progress = batch * 1.0 / len(self.val_dataloader) * progress_bar_length
            
            for i in range (0,  math.floor(progress), 1): 
                print (progress_char, end = "")
            for i in range (0,  progress_bar_length - math.floor(progress), 1): 
                if (i == 0):
                    print (math.floor(10 * (progress - math.floor(progress))), end="")
                else: 
                    print (remaining_char, end = "")
            print (f" {batch} out of {len(self.val_dataloader)}")
        
        if (epoch != 0): 
            print (f"Latest train loss: {self.train_loss_log[epoch - 1]}")
            print (f"Latest val accuracy: {self.accuracy_log[epoch - 1]}")
        
        # plt.clf()
        # x-axis: epoch
        # y-axis: train loss & val accuracy
        loss = []
        accuracy = []
        for i in self.train_loss_log: 
            loss.append(round(i, 2))
        for i in self.accuracy_log: 
            accuracy.append(round(i, 4))
        
        fig, plot1 = plt.subplots (figsize = (10, 8))
        plt.title (f"Model {self.model_name}'s train loss & val accuracy")
        
        plot1.set_xlim (0, self.train_model_epoch + 1)
        if (len(self.train_loss_log) != 0): 
            plot1.set_ylim (0, max(self.train_loss_log) * 1.2)
        else: plot1.set_ylim (0, 1000000)
        plot1.set_xlabel ("Epoch")
        plot1.set_ylabel ("Loss")
        plot1_x = list(range(1, len(self.train_loss_log) + 1))
        plot1_y = self.train_loss_log
        plot1_legend, = plot1.plot(plot1_x, plot1_y, color = 'r', label = "Loss")
        
        plot2 = plot1.twinx()
        plot2.set_xlim (0, self.train_model_epoch + 1)
        plot2.set_ylim (0, 1.2)
        plot2.set_xlabel ("Epoch")
        plot2.set_ylabel ("Accuracy")
        plot2_x = list(range(1, len(self.accuracy_log) + 1))
        plot2_y = self.accuracy_log
        plot2_legend, = plot2.plot(plot2_x, plot2_y, color = 'g', label = "Accuracy")
        
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
        legends=[plot1_legend, plot2_legend]
        plot1.legend(handles=legends, loc=1)
        
        
    
    def train (self, setup, dataset): 
        
        self.train_loss_log = []
        self.accuracy_log = []
        
        clear_output(wait=True)
        print (f"Loading shuffled train and val dataset from {dataset.dataset}")
        self.train_dataloader = torch.utils.data.DataLoader (dataset = dataset.train_dataset, batch_size = self.train_model_batch_size, shuffle = True)
        self.val_dataloader = torch.utils.data.DataLoader (dataset = dataset.val_dataset, batch_size = self.train_model_batch_size, shuffle = True)    

        for epoch in range (self.train_model_epoch): 
            
            # Training
            self.model.train()
            training_loss_total = 0
            for batch, data in enumerate(self.train_dataloader, start = 0): 
                
                images, labels = data[0].to(self.device), data[1].to(self.device)
                self.model_optimizer.zero_grad()
                
                train_prediction = self.model(images)
                # print (train_prediction)
                
                _, train_prediction_class = torch.max(train_prediction.data, dim=1)
                
                print ("Prediction:", train_prediction_class)
                print ("Labels:", labels)
                # input ()
                
                training_loss = 0
                if (self.model_name == "Inception_V1"): 
                    training_loss = self.model_loss_function(training_prediction[0], labels)
                    + 0.3 * self.model_loss_function(training_prediction[1], labels)
                    + 0.3 * self.model_loss_function(training_prediction[2], labels)    
                else: 
                    training_loss = self.model_loss_function(train_prediction, labels)
                    
                training_loss.backward()
                self.model_optimizer.step()
                training_loss_total += training_loss.item()
                
                self.print_train_val_plot ("train", epoch, batch)
                
            self.train_loss_log.append(training_loss_total)
            
            
            # Validation
            self.model.eval()
            correct_predictions = 0
            total_predictions = 0
            
            with torch.no_grad():
                
                for batch, data in enumerate(self.val_dataloader, start = 0): 
                    
                    images, labels = data[0].to(self.device), data[1].to(self.device)
                    val_prediction = self.model(images)
                    
                    # output_softmax = [nn.functional.softmax(i, dim=0) for i in output]
                    
                    # print(val_prediction)
                    _, val_prediction_class = torch.max(val_prediction.data, dim=1)
                    correct_predictions += (val_prediction_class == labels).sum().item()
                    
                    print ("Prediction:", val_prediction_class)
                    print ("Labels:", labels)
                    
                    total_predictions += labels.size(0)
                    
                    # input()
                 
                    self.print_train_val_plot ("val", epoch, batch)

            accuracy = correct_predictions / total_predictions
            print (correct_predictions, total_predictions, accuracy)
            self.accuracy_log.append(accuracy)
            
            
        self.print_train_val_plot ("finish", None, None)
        
