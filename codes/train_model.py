
import os
from IPython.display import clear_output
import math

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

class train_model:
    
    def __init__ (self, setup, dataset): 
        
        self.root = setup.path
        clear_output(wait=True)
        
        file_read = open(f"{self.root}/configurations/model.txt")
        for line in file_read.readlines():
            if (len(line) > 7 and line[0:7] == "[model]"): self.model_name = line[7:len(line) - 1]
            if (len(line) > 15 and line[0:15] == "[loss_function]"): self.model_loss_function_name = line[15:len(line) - 1]
            if (len(line) > 11 and line[0:11] == "[optimizer]"): self.model_optimizer_name = line[11:len(line) - 1]
            if (len(line) > 15 and line[0:15] == "[learning_rate]"): self.model_optimizer_lr = line[15:len(line) - 1]
            if (len(line) > 10 and line[0:10] == "[momentum]"): self.model_optimizer_momentum = line[10:len(line) - 1]
            if (len(line) > 9 and line[0:9] == "[epoches]"): self.train_model_epoch = line[9:len(line) - 1]
            if (len(line) > 12 and line[0:12] == "[batch_size]"): self.train_model_batch_size = line[12:len(line) - 1]
        file_read.close()
        
        all_model = [f.name for f in os.scandir(f"{self.root}/models") if f.is_file() and len(f.name) > 3 and f.name[len(f.name)-3:len(f.name)] == ".py"]
        for i in range (len(all_model)): 
            all_model[i] = all_model[i][0:len(all_model[i]) - 3]
        all_model = sorted(all_model)
        
        print (f"Model: {self.model_name}")
        if (self.model_name not in all_model): 
            print (f"All models: {all_model}")
            raise Exception (f"[Error] {self.model_name} does not exist in the models folder. ")
        exec(f"import models.{self.model_name} as {self.model_name}")
        self.model = eval(f"{self.model_name}.{self.model_name} ()")
        if (self.model == None):
            raise Exception (f"[Error] Generating the model {self.model_name} failed. ")
        
        print (f"Model loss function: {self.model_loss_function_name}")
        all_model_loss_function = ["CrossEntropyLoss"]
        if (self.model_loss_function_name not in all_model_loss_function): 
            print (f"All model loss functions: {all_model_loss_function}")
            raise Exception (f"[Error] {self.model_loss_function_name} is not a valid model loss function. ")
        self.model_loss_function = eval(f"nn.{self.model_loss_function_name}()")

        print (f"Model optimizer: {self.model_optimizer_name}")
        all_model_optimizer = ["Adam", "SGD"]
        if (self.model_optimizer_name not in all_model_optimizer): 
            print (f"All model optimizers: {all_model_optimizer}")
            raise Exception (f"[Error] {self.model_optimizer_name} is not a valid model optimizer. ")
        
        print (f"Model optimizer learning rate: {self.model_optimizer_lr}")
        try: 
            self.model_optimizer_lr = float(self.model_optimizer_lr)
            if (not (0 < self.model_optimizer_lr)): raise Exception (f"[Error] Optimizer learning rate should be a positive float number. ")
        except ValueError: raise Exception (f"[Error] Optimizer learning rate should be a positive float number. ")
        
        print (f"Model optimizer momentum: {self.model_optimizer_momentum}")
        if (self.model_optimizer_name == "SGD"): 
            try: 
                self.model_optimizer_momentum = float(self.model_optimizer_momentum)
                if (not (0 <= self.model_optimizer_momentum and self.model_optimizer_momentum < 1)): 
                    raise Exception (f"[Error] Optimizer momentum should be a positive float number between [0, 1). ")
            except ValueError: raise Exception (f"[Error] Optimizer momentum should be a positive float number between [0, 1). ")
        else: self.model_optimizer_momentum = None
        import torch.optim as optim
        if (self.model_optimizer_name == "SGD"): 
            self.model_optimizer = eval(f"optim.{self.model_optimizer_name}(self.model.parameters(), lr=self.model_optimizer_lr, momentum={self.model_optimizer_momentum})")
        else: self.model_optimizer = eval(f"optim.{self.model_optimizer_name}(self.model.parameters(), lr=self.model_optimizer_lr)")
        
        
        print (f"Model train epoches: {self.train_model_epoch}")
        try: 
            self.train_model_epoch = int(self.train_model_epoch)
            if (not (0 < self.train_model_epoch)): raise Exception (f"[Error] Model train epoches should be a positive integer. ")
        except ValueError: raise Exception (f"[Error] Model train epoches should be a positive integer. ")

        print (f"Model batch size: {self.train_model_batch_size}")
        try: 
            self.train_model_batch_size = int(self.train_model_batch_size)
            if (not (0 < self.train_model_batch_size)): raise Exception (f"[Error] Model train batch size should be a positive integer. ")
        except ValueError: raise Exception (f"[Error] Model train batch size should be a positive integer. ")            
            
               
        clear_output(wait=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if (self.device == "cpu"): print("[Device] Warning: Train the model with CPU because cuda is not available")
        else: print("[Device] Train the model with GPU")
        self.model.to(self.device)
                
        clear_output(wait=True)
        print ("All model training settings are completed. ")
        print (f"Model name: {self.model_name} (output classes = 17)")
        print (f"Loss function: {self.model_loss_function_name}")
        print (f"Optimizer: {self.model_optimizer_name} (learning rate={self.model_optimizer_lr}, momentum={self.model_optimizer_momentum})")
        print ("\nFor the model training: ")
        print (f"Epoches: {self.train_model_epoch}")
        print (f"Batch size: {self.train_model_batch_size}")
        if torch.cuda.is_available(): print (f"Device: GPU")
        else: print (f"Device: CPU")
        print ("\nThe training will start in 5 seconds ...")
        import time
        time.sleep (5)
        self.train(setup, dataset)
        
        self.print_train_result(setup)
        
        self.save_trained_model(setup, dataset)

            
            
    def print_train_val (self, option, epoch, batch): 
        
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
        
        if (option != "finish"): 
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
                _, train_prediction_class = torch.max(train_prediction.data, dim=1)
                # print ("Prediction:", train_prediction_class)
                # print ("Labels:", labels)
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
                self.print_train_val ("train", epoch, batch)
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
                    # print ("Prediction:", val_prediction_class)
                    # print ("Labels:", labels)
                    total_predictions += labels.size(0)
                    self.print_train_val ("val", epoch, batch)

            accuracy = correct_predictions / total_predictions
            self.accuracy_log.append(accuracy)
            
            
        self.print_train_val ("finish", self.train_model_epoch, None)
        

    def print_train_result(self, setup): 
        
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
        
        plt.savefig(f"{setup.path}/figures/[Train] model={self.model_name}  loss_function={self.model_loss_function_name}  optimizer={self.model_optimizer_name}  learning_rate={self.model_optimizer_lr}  momentum={self.model_optimizer_momentum}  epoches={self.train_model_epoch}  batch_size={self.train_model_batch_size}.pdf", format="pdf", bbox_inches="tight")
        
    def save_trained_model(self, setup, dataset):
        
        pass