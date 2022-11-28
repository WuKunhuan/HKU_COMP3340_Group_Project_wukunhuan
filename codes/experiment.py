
import os
import torch
import matplotlib
import matplotlib.pyplot as plt

try: 
    os.system ("pip install sklearn")
    print ("[Package] Download sklearn package successfully. ")
except Exception: print ("[Error] Failed to download the sklearn package. ")
from sklearn.metrics import confusion_matrix

import numpy as np

try: 
    os.system ("pip install seaborn")
    print ("[Package] Download seaborn package successfully. ")
except Exception: print ("[Error] Failed to download the seaborn package. ")
import seaborn as sn

try: 
    os.system ("pip install pandas")
    print ("[Package] Download pandas package successfully. ")
except Exception: print ("[Error] Failed to download the pandas package. ")
import pandas as pd



# https://chartio.com/learn/charts/how-to-choose-colors-data-visualization/
# https://stackoverflow.com/questions/57043260/how-change-the-color-of-boxes-in-confusion-matrix-using-sklearn

# Compare different models
color_schema_1 = [(40/255,86/255,108/255), (182/255,121/255,144/255), (118/255,181/255,106/255), (187/255,80/255,56/255), (106/255,79/255,121/255), (143/255,198/255,189/255)]

# Compare different learning rate & batch sizes
color_schema_r = [(0.5, 0, 0), (0.65, 0.3, 0.3), (0.75, 0.5, 0.5), (0.82, 0.65, 0.65), (0.87, 0.75, 0.75), (0.91, 0.82, 0.82)]
color_schema_g = [(0, 0.5, 0), (0.3, 0.65, 0.3), (0.5, 0.75, 0.5), (0.65, 0.82, 0.65), (0.75, 0.87, 0.75), (0.82, 0.91, 0.82)]
color_schema_b = [(0, 0, 0.5), (0.3, 0.3, 0.65), (0.5, 0.5, 0.75), (0.65, 0.65, 0.82), (0.75, 0.75, 0.87), (0.82, 0.82, 0.91)]

# Combine two dimensions
dash_schema = [None, [6, 2], [3, 2], [1, 1], [0.5, 0.5]]







class Experiment: 
    
    def __init__(self, *args): 
        self.models = args
        self.model_name = None
        self.dataset = None
        self.model_loss_function_name = None
        self.model_optimizer_name = None
        self.model_optimizer_lr = None
        self.model_optimizer_momentum = None
        self.train_model_epoch = None
        self.train_model_batch_size = None
        self.train_loss_log = []
        self.train_accuracy_log = []
        self.val_accuracy_log = []
        self.train_time = []
        self.train_time_epoch = []
        
        
        
    def model_trainloss(self, setup, color_schema_1):
        
        self.save_model_name = []
        
        self.train_loss_log = []
        
        warning = False
        
        print ("[Experiment] Find the relationship between models and Train Losses & Validation Accuracies (top 1 & top 5). \n")
        
        for i in range (len(self.models)): 
            
            if (self.models[i].save_model_name in self.save_model_name): 
                print (f"Warning: {self.models[i].save_model_name} already exists! ")
                warning = True
            self.save_model_name.append(self.models[i].save_model_name)
            
            self.train_loss_log.append(self.models[i].train_loss_log)
            
            if (i == 0): 
                self.dataset = self.models[i].dataset
                self.model_loss_function_name = self.models[i].model_loss_function_name
                self.model_optimizer_name = self.models[i].model_optimizer_name
                self.model_optimizer_lr = self.models[i].model_optimizer_lr
                self.model_optimizer_momentum = self.models[i].model_optimizer_momentum
                self.train_model_epoch = self.models[i].train_model_epoch
                self.train_model_batch_size = self.models[i].train_model_batch_size
            else:
                if (self.models[i].dataset != self.dataset): 
                    print (f"Warning: {self.models[i].save_model_name}'s dataset ({self.models[i].dataset}) is different from  {self.models[0].save_model_name}'s dataset ({self.dataset})! ")
                    warning = True
                if (self.models[i].model_loss_function_name != self.model_loss_function_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s loss function ({self.models[i].model_loss_function_name}) is different from  {self.models[0].save_model_name}'s loss function ({self.model_loss_function_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_name != self.model_optimizer_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer ({self.models[i].model_optimizer_name}) is different from  {self.models[0].save_model_name}'s optimizer ({self.model_optimizer_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_lr != self.model_optimizer_lr): 
                    print (f"Warning: {self.models[i].save_model_name}'s learning rate ({self.models[i].model_optimizer_lr}) is different from  {self.models[0].save_model_name}'s learning rate ({self.model_optimizer_lr})! ")
                    warning = True
                if (self.models[i].model_optimizer_momentum != self.model_optimizer_momentum): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer momentum ({self.models[i].model_optimizer_momentum}) is different from  {self.models[0].save_model_name}'s optimizer momentum ({self.model_optimizer_momentum})! ")
                    warning = True
                if (self.models[i].train_model_epoch != self.train_model_epoch): 
                    print (f"Warning: {self.models[i].save_model_name}'s train epoches ({self.models[i].train_model_epoch}) is different from  {self.models[0].save_model_name}'s train epoches ({self.train_model_epoch})! ")
                    warning = True
                if (self.models[i].train_model_batch_size != self.train_model_batch_size): 
                    print (f"Warning: {self.models[i].save_model_name}'s batch size ({self.models[i].train_model_batch_size}) is different from  {self.models[0].save_model_name}'s batch size ({self.train_model_batch_size})! ")
                    warning = True

        if (warning): 
            i = input("\nConsider the above warning(s), press Enter to discard or enter anything to continue drawing the chart. ")
            if (i == ""): 
                print ("User has quitted drawing the chart. ")
                return; 
        
        print ("[Controlled Variables]")
        print (f"dataset: {self.dataset}")
        print (f"loss function: {self.model_loss_function_name}")
        print (f"optimizer: {self.model_optimizer_name} (momentum = {self.model_optimizer_momentum})")
        print (f"learning rate: {self.model_optimizer_lr}")
        print (f"epoch: {self.train_model_epoch}")
        print (f"batch size: {self.train_model_batch_size}\n")
        print (f"train loss: {self.train_loss_log}")
            
        fig, plot = plt.subplots (figsize = (10, 8))
        
        plt.title (f"Train Losses for different Models")
        plt.yticks([]) 
        legends = []
        
        max_train_loss = 0
        for i in self.train_loss_log: 
            if (max(i) > max_train_loss): max_train_loss = max(i)
        
        
        for i in range (len(self.models)):
            
            exec(f"plot{3*i} = plot.twinx()")
            exec(f"plot{3*i}.set_xlim (0, {self.train_model_epoch} + 1)")
            exec(f"plot{3*i}.set_xlabel (\"Epoch\")")
            exec(f"plot{3*i}.set_ylim (0, max_train_loss*1.2)")
            exec(f"plot{3*i}.set_ylabel (\"Train Loss\")")
            exec(f"plot{3*i}_x = list(range(1, self.train_model_epoch + 1))")
            exec(f"plot{3*i}_y = self.train_loss_log[i]")
            exec(f"plot{3*i}_legend, = plot{3*i}.plot(plot{3*i}_x, plot{3*i}_y, color = {eval(color_schema_1)[i]}, label = \"{self.save_model_name[i]} train.loss\")")
            legends.append(eval(f"plot{3*i}_legend"))
            
        plot.legend(handles=legends, bbox_to_anchor=(-0.03, 1))
        
        i = 1; 
        while (os.path.exists(f"{setup.path}/figures/[Experiment] model_trainloss ({i}).pdf")): i += 1
        plt.savefig(f"{setup.path}/figures/[Experiment] model_trainloss ({i}).pdf", format="pdf", bbox_inches="tight")
        print (f"\n[Figure] The below figure can be found in \"figures/[Experiment] model_trainloss ({i}).pdf\"")
        
        


    def model_trainacc1(self, setup, color_schema_1): 

        self.save_model_name = []
        
        self.train_accuracy_log = []
        
        warning = False

        
        print ("[Experiment] Find the relationship between Models and Validation Accuracies (top 1 & top 5). \n")
        
        for i in range (len(self.models)): 
            
            if (self.models[i].save_model_name in self.save_model_name): 
                print (f"Warning: {self.models[i].save_model_name} already exists! ")
                warning = True
            self.save_model_name.append(self.models[i].save_model_name)
            
            self.train_accuracy_log.append(self.models[i].train_accuracy_log)
            
            if (i == 0): 
                self.dataset = self.models[i].dataset
                self.model_loss_function_name = self.models[i].model_loss_function_name
                self.model_optimizer_name = self.models[i].model_optimizer_name
                self.model_optimizer_lr = self.models[i].model_optimizer_lr
                self.model_optimizer_momentum = self.models[i].model_optimizer_momentum
                self.train_model_epoch = self.models[i].train_model_epoch
                self.train_model_batch_size = self.models[i].train_model_batch_size
            else:
                if (self.models[i].dataset != self.dataset): 
                    print (f"Warning: {self.models[i].save_model_name}'s dataset ({self.models[i].dataset}) is different from  {self.models[0].save_model_name}'s dataset ({self.dataset})! ")
                    warning = True
                if (self.models[i].model_loss_function_name != self.model_loss_function_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s loss function ({self.models[i].model_loss_function_name}) is different from  {self.models[0].save_model_name}'s loss function ({self.model_loss_function_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_name != self.model_optimizer_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer ({self.models[i].model_optimizer_name}) is different from  {self.models[0].save_model_name}'s optimizer ({self.model_optimizer_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_lr != self.model_optimizer_lr): 
                    print (f"Warning: {self.models[i].save_model_name}'s learning rate ({self.models[i].model_optimizer_lr}) is different from  {self.models[0].save_model_name}'s learning rate ({self.model_optimizer_lr})! ")
                    warning = True
                if (self.models[i].model_optimizer_momentum != self.model_optimizer_momentum): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer momentum ({self.models[i].model_optimizer_momentum}) is different from  {self.models[0].save_model_name}'s optimizer momentum ({self.model_optimizer_momentum})! ")
                    warning = True
                if (self.models[i].train_model_epoch != self.train_model_epoch): 
                    print (f"Warning: {self.models[i].save_model_name}'s train epoches ({self.models[i].train_model_epoch}) is different from  {self.models[0].save_model_name}'s train epoches ({self.train_model_epoch})! ")
                    warning = True
                if (self.models[i].train_model_batch_size != self.train_model_batch_size): 
                    print (f"Warning: {self.models[i].save_model_name}'s batch size ({self.models[i].train_model_batch_size}) is different from  {self.models[0].save_model_name}'s batch size ({self.train_model_batch_size})! ")
                    warning = True

        if (warning): 
            i = input("\nConsider the above warning(s), press Enter to discard or enter anything to continue drawing the chart. ")
            if (i == ""): 
                print ("User has quitted drawing the chart. ")
                return; 
        
        print ("[Controlled Variables]")
        print (f"dataset: {self.dataset}")
        print (f"loss function: {self.model_loss_function_name}")
        print (f"optimizer: {self.model_optimizer_name} (momentum = {self.model_optimizer_momentum})")
        print (f"learning rate: {self.model_optimizer_lr}")
        print (f"epoch: {self.train_model_epoch}")
        print (f"batch size: {self.train_model_batch_size}\n")
        print (f"train accuracy (top 1): {self.train_accuracy_log}")
            
        fig, plot = plt.subplots (figsize = (10, 8))
        
        plt.title (f"Train Accuracies (top1) for different Models")
        plt.yticks([]) 
        legends = []
        
        max_train_loss = 0
        for i in self.train_loss_log: 
            if (max(i) > max_train_loss): max_train_loss = max(i)
        
        
        for i in range (len(self.models)):

            exec(f"plot{3*i+1} = plot.twinx()")
            exec(f"plot{3*i+1}.set_xlim (0, {self.train_model_epoch} + 1)")
            exec(f"plot{3*i+1}.set_xlabel (\"Epoch\")")
            exec(f"plot{3*i+1}.set_ylim (0, 1)")
            exec(f"plot{3*i+1}.set_ylabel (\"Train Accuracy (top 1)\")")
            exec(f"plot{3*i+1}_x = list(range(1, self.train_model_epoch + 1))")
            exec(f"plot{3*i+1}_y = self.train_accuracy_log[i]")
            exec(f"plot{3*i+1}_legend, = plot{3*i+1}.plot(plot{3*i+1}_x, plot{3*i+1}_y, color = {eval(color_schema_1)[i]}, label = \"{self.save_model_name[i]} val.acc1\")")
            legends.append(eval(f"plot{3*i+1}_legend"))

        plot.legend(handles=legends, bbox_to_anchor=(-0.03, 1))
        
        i = 1; 
        while (os.path.exists(f"{setup.path}/figures/[Experiment] model_trainacc1 ({i}).pdf")): i += 1
        plt.savefig(f"{setup.path}/figures/[Experiment] model_trainacc1 ({i}).pdf", format="pdf", bbox_inches="tight")
        print (f"\n[Figure] The below figure can be found in \"figures/[Experiment] model_trainacc1 ({i}).pdf\"")  
        
    
        
    def model_valacc1(self, setup, color_schema_1): 

        self.save_model_name = []
        
        self.val_accuracy_log = []
        
        warning = False

        
        print ("[Experiment] Find the relationship between Models and Validation Accuracies (top 1 & top 5). \n")
        
        for i in range (len(self.models)): 
            
            if (self.models[i].save_model_name in self.save_model_name): 
                print (f"Warning: {self.models[i].save_model_name} already exists! ")
                warning = True
            self.save_model_name.append(self.models[i].save_model_name)
            
            self.val_accuracy_log.append(self.models[i].val_accuracy_log)
            
            if (i == 0): 
                self.dataset = self.models[i].dataset
                self.model_loss_function_name = self.models[i].model_loss_function_name
                self.model_optimizer_name = self.models[i].model_optimizer_name
                self.model_optimizer_lr = self.models[i].model_optimizer_lr
                self.model_optimizer_momentum = self.models[i].model_optimizer_momentum
                self.train_model_epoch = self.models[i].train_model_epoch
                self.train_model_batch_size = self.models[i].train_model_batch_size
            else:
                if (self.models[i].dataset != self.dataset): 
                    print (f"Warning: {self.models[i].save_model_name}'s dataset ({self.models[i].dataset}) is different from  {self.models[0].save_model_name}'s dataset ({self.dataset})! ")
                    warning = True
                if (self.models[i].model_loss_function_name != self.model_loss_function_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s loss function ({self.models[i].model_loss_function_name}) is different from  {self.models[0].save_model_name}'s loss function ({self.model_loss_function_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_name != self.model_optimizer_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer ({self.models[i].model_optimizer_name}) is different from  {self.models[0].save_model_name}'s optimizer ({self.model_optimizer_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_lr != self.model_optimizer_lr): 
                    print (f"Warning: {self.models[i].save_model_name}'s learning rate ({self.models[i].model_optimizer_lr}) is different from  {self.models[0].save_model_name}'s learning rate ({self.model_optimizer_lr})! ")
                    warning = True
                if (self.models[i].model_optimizer_momentum != self.model_optimizer_momentum): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer momentum ({self.models[i].model_optimizer_momentum}) is different from  {self.models[0].save_model_name}'s optimizer momentum ({self.model_optimizer_momentum})! ")
                    warning = True
                if (self.models[i].train_model_epoch != self.train_model_epoch): 
                    print (f"Warning: {self.models[i].save_model_name}'s train epoches ({self.models[i].train_model_epoch}) is different from  {self.models[0].save_model_name}'s train epoches ({self.train_model_epoch})! ")
                    warning = True
                if (self.models[i].train_model_batch_size != self.train_model_batch_size): 
                    print (f"Warning: {self.models[i].save_model_name}'s batch size ({self.models[i].train_model_batch_size}) is different from  {self.models[0].save_model_name}'s batch size ({self.train_model_batch_size})! ")
                    warning = True

        if (warning): 
            i = input("\nConsider the above warning(s), press Enter to discard or enter anything to continue drawing the chart. ")
            if (i == ""): 
                print ("User has quitted drawing the chart. ")
                return; 
        
        print ("[Controlled Variables]")
        print (f"dataset: {self.dataset}")
        print (f"loss function: {self.model_loss_function_name}")
        print (f"optimizer: {self.model_optimizer_name} (momentum = {self.model_optimizer_momentum})")
        print (f"learning rate: {self.model_optimizer_lr}")
        print (f"epoch: {self.train_model_epoch}")
        print (f"batch size: {self.train_model_batch_size}\n")
        print (f"validation accuracy (top 1): {self.val_accuracy_log}")
            
        fig, plot = plt.subplots (figsize = (10, 8))
        
        plt.title (f"Validation Accuracies (top1) for different Models")
        plt.yticks([]) 
        legends = []
        
        max_train_loss = 0
        for i in self.train_loss_log: 
            if (max(i) > max_train_loss): max_train_loss = max(i)
        
        
        for i in range (len(self.models)):

            exec(f"plot{3*i+1} = plot.twinx()")
            exec(f"plot{3*i+1}.set_xlim (0, {self.train_model_epoch} + 1)")
            exec(f"plot{3*i+1}.set_xlabel (\"Epoch\")")
            exec(f"plot{3*i+1}.set_ylim (0, 1)")
            exec(f"plot{3*i+1}.set_ylabel (\"Validation Accuracy (top 1)\")")
            exec(f"plot{3*i+1}_x = list(range(1, self.train_model_epoch + 1))")
            exec(f"plot{3*i+1}_y = self.val_accuracy_log[i]")
            exec(f"plot{3*i+1}_legend, = plot{3*i+1}.plot(plot{3*i+1}_x, plot{3*i+1}_y, color = {eval(color_schema_1)[i]}, label = \"{self.save_model_name[i]} val.acc1\")")
            legends.append(eval(f"plot{3*i+1}_legend"))

        plot.legend(handles=legends, bbox_to_anchor=(-0.03, 1))
        
        i = 1; 
        while (os.path.exists(f"{setup.path}/figures/[Experiment] model_valacc1 ({i}).pdf")): i += 1
        plt.savefig(f"{setup.path}/figures/[Experiment] model_valacc1 ({i}).pdf", format="pdf", bbox_inches="tight")
        print (f"\n[Figure] The below figure can be found in \"figures/[Experiment] model_valacc1 ({i}).pdf\"")       

        
        
        
    def model_trainlossvalacc1(self, setup, color_schema_1, color_schema_2):
        
        self.save_model_name = []
        
        self.train_loss_log = []
        self.val_accuracy_log = []
        
        warning = False

        
        print ("[Experiment] Find the relationship between Models and Train Losses & Validation Accuracies (top 1 & top 5). \n")
        
        for i in range (len(self.models)): 
            
            if (self.models[i].save_model_name in self.save_model_name): 
                print (f"Warning: {self.models[i].save_model_name} already exists! ")
                warning = True
            self.save_model_name.append(self.models[i].save_model_name)
            
            self.train_loss_log.append(self.models[i].train_loss_log)
            self.val_accuracy_log.append(self.models[i].val_accuracy_log)
            
            if (i == 0): 
                self.dataset = self.models[i].dataset
                self.model_loss_function_name = self.models[i].model_loss_function_name
                self.model_optimizer_name = self.models[i].model_optimizer_name
                self.model_optimizer_lr = self.models[i].model_optimizer_lr
                self.model_optimizer_momentum = self.models[i].model_optimizer_momentum
                self.train_model_epoch = self.models[i].train_model_epoch
                self.train_model_batch_size = self.models[i].train_model_batch_size
            else:
                if (self.models[i].dataset != self.dataset): 
                    print (f"Warning: {self.models[i].save_model_name}'s dataset ({self.models[i].dataset}) is different from  {self.models[0].save_model_name}'s dataset ({self.dataset})! ")
                    warning = True
                if (self.models[i].model_loss_function_name != self.model_loss_function_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s loss function ({self.models[i].model_loss_function_name}) is different from  {self.models[0].save_model_name}'s loss function ({self.model_loss_function_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_name != self.model_optimizer_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer ({self.models[i].model_optimizer_name}) is different from  {self.models[0].save_model_name}'s optimizer ({self.model_optimizer_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_lr != self.model_optimizer_lr): 
                    print (f"Warning: {self.models[i].save_model_name}'s learning rate ({self.models[i].model_optimizer_lr}) is different from  {self.models[0].save_model_name}'s learning rate ({self.model_optimizer_lr})! ")
                    warning = True
                if (self.models[i].model_optimizer_momentum != self.model_optimizer_momentum): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer momentum ({self.models[i].model_optimizer_momentum}) is different from  {self.models[0].save_model_name}'s optimizer momentum ({self.model_optimizer_momentum})! ")
                    warning = True
                if (self.models[i].train_model_epoch != self.train_model_epoch): 
                    print (f"Warning: {self.models[i].save_model_name}'s train epoches ({self.models[i].train_model_epoch}) is different from  {self.models[0].save_model_name}'s train epoches ({self.train_model_epoch})! ")
                    warning = True
                if (self.models[i].train_model_batch_size != self.train_model_batch_size): 
                    print (f"Warning: {self.models[i].save_model_name}'s batch size ({self.models[i].train_model_batch_size}) is different from  {self.models[0].save_model_name}'s batch size ({self.train_model_batch_size})! ")
                    warning = True

        if (warning): 
            i = input("\nConsider the above warning(s), press Enter to discard or enter anything to continue drawing the chart. ")
            if (i == ""): 
                print ("User has quitted drawing the chart. ")
                return; 
        
        print ("[Controlled Variables]")
        print (f"dataset: {self.dataset}")
        print (f"loss function: {self.model_loss_function_name}")
        print (f"optimizer: {self.model_optimizer_name} (momentum = {self.model_optimizer_momentum})")
        print (f"learning rate: {self.model_optimizer_lr}")
        print (f"epoch: {self.train_model_epoch}")
        print (f"batch size: {self.train_model_batch_size}\n")
        print (f"train loss: {self.train_loss_log}")
        print (f"validation accuracy (top 1): {self.val_accuracy_log}")
            
        fig, plot = plt.subplots (figsize = (10, 8))
        
        plt.title (f"Train Losses and Validation Accuracies (top1) for different models")
        plt.yticks([]) 
        legends = []
        
        max_train_loss = 0
        for i in self.train_loss_log: 
            if (max(i) > max_train_loss): max_train_loss = max(i)
        
        
        for i in range (len(self.models)):
            
            exec(f"plot{3*i} = plot.twinx()")
            exec(f"plot{3*i}.set_xlim (0, {self.train_model_epoch} + 1)")
            exec(f"plot{3*i}.set_xlabel (\"Epoch\")")
            exec(f"plot{3*i}.set_ylim (0, max_train_loss*1.2)")
            exec(f"plot{3*i}.set_ylabel (\"Train Loss\")")
            exec(f"plot{3*i}_x = list(range(1, self.train_model_epoch + 1))")
            exec(f"plot{3*i}_y = self.train_loss_log[i]")
            exec(f"plot{3*i}_legend, = plot{3*i}.plot(plot{3*i}_x, plot{3*i}_y, color = {eval(color_schema_1)[i]}, label = \"{self.save_model_name[i]} train.loss\")")
            legends.append(eval(f"plot{3*i}_legend"))
            
            exec(f"plot{3*i+1} = plot.twinx()")
            exec(f"plot{3*i+1}.spines['right'].set_position(('outward', 60))")
            exec(f"plot{3*i+1}.set_xlim (0, {self.train_model_epoch} + 1)")
            exec(f"plot{3*i+1}.set_xlabel (\"Epoch\")")
            exec(f"plot{3*i+1}.set_ylim (0, 1)")
            exec(f"plot{3*i+1}.set_ylabel (\"Validation Accuracy (top 1)\")")
            exec(f"plot{3*i+1}_x = list(range(1, self.train_model_epoch + 1))")
            exec(f"plot{3*i+1}_y = self.val_accuracy_log[i]")
            exec(f"plot{3*i+1}_legend, = plot{3*i+1}.plot(plot{3*i+1}_x, plot{3*i+1}_y, color = {eval(color_schema_2)[i]}, label = \"{self.save_model_name[i]} val.acc1\")")
            legends.append(eval(f"plot{3*i+1}_legend"))
                    
            
        plot.legend(handles=legends, bbox_to_anchor=(-0.03, 1))
        
        i = 1; 
        while (os.path.exists(f"{setup.path}/figures/[Experiment] model_trainlossvalacc1 ({i}).pdf")): i += 1
        plt.savefig(f"{setup.path}/figures/[Experiment] model_trainlossvalacc1 ({i}).pdf", format="pdf", bbox_inches="tight")
        print (f"\n[Figure] The below figure can be found in \"figures/[Experiment] model_trainlossvalacc1 ({i}).pdf\"")        
        
        
        
    
    def model_trainlossvalacc1valacc5(self, setup, color_schema_1, color_schema_2, color_schema_3): 
        
        self.save_model_name = []
        
        self.train_loss_log = []
        self.val_accuracy_log = []
        self.val_accuracy_5_log = []
        
        warning = False

        print ("[Experiment] Find the relationship between Models and Train Losses & Validation Accuracies (top 1 & top 5). \n")
        
        for i in range (len(self.models)): 
            
            if (self.models[i].save_model_name in self.save_model_name): 
                print (f"Warning: {self.models[i].save_model_name} already exists! ")
                warning = True
            self.save_model_name.append(self.models[i].save_model_name)
            
            self.train_loss_log.append(self.models[i].train_loss_log)
            self.val_accuracy_log.append(self.models[i].val_accuracy_log)
            self.val_accuracy_5_log.append(self.models[i].val_accuracy_5_log)
            
            if (i == 0): 
                self.dataset = self.models[i].dataset
                self.model_loss_function_name = self.models[i].model_loss_function_name
                self.model_optimizer_name = self.models[i].model_optimizer_name
                self.model_optimizer_lr = self.models[i].model_optimizer_lr
                self.model_optimizer_momentum = self.models[i].model_optimizer_momentum
                self.train_model_epoch = self.models[i].train_model_epoch
                self.train_model_batch_size = self.models[i].train_model_batch_size
            else:
                if (self.models[i].dataset != self.dataset): 
                    print (f"Warning: {self.models[i].save_model_name}'s dataset ({self.models[i].dataset}) is different from  {self.models[0].save_model_name}'s dataset ({self.dataset})! ")
                    warning = True
                if (self.models[i].model_loss_function_name != self.model_loss_function_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s loss function ({self.models[i].model_loss_function_name}) is different from  {self.models[0].save_model_name}'s loss function ({self.model_loss_function_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_name != self.model_optimizer_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer ({self.models[i].model_optimizer_name}) is different from  {self.models[0].save_model_name}'s optimizer ({self.model_optimizer_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_lr != self.model_optimizer_lr): 
                    print (f"Warning: {self.models[i].save_model_name}'s learning rate ({self.models[i].model_optimizer_lr}) is different from  {self.models[0].save_model_name}'s learning rate ({self.model_optimizer_lr})! ")
                    warning = True
                if (self.models[i].model_optimizer_momentum != self.model_optimizer_momentum): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer momentum ({self.models[i].model_optimizer_momentum}) is different from  {self.models[0].save_model_name}'s optimizer momentum ({self.model_optimizer_momentum})! ")
                    warning = True
                if (self.models[i].train_model_epoch != self.train_model_epoch): 
                    print (f"Warning: {self.models[i].save_model_name}'s train epoches ({self.models[i].train_model_epoch}) is different from  {self.models[0].save_model_name}'s train epoches ({self.train_model_epoch})! ")
                    warning = True
                if (self.models[i].train_model_batch_size != self.train_model_batch_size): 
                    print (f"Warning: {self.models[i].save_model_name}'s batch size ({self.models[i].train_model_batch_size}) is different from  {self.models[0].save_model_name}'s batch size ({self.train_model_batch_size})! ")
                    warning = True

        if (warning): 
            i = input("\nConsider the above warning(s), press Enter to discard or enter anything to continue drawing the chart. ")
            if (i == ""): 
                print ("User has quitted drawing the chart. ")
                return; 
        
        print ("[Controlled Variables]")
        print (f"dataset: {self.dataset}")
        print (f"loss function: {self.model_loss_function_name}")
        print (f"optimizer: {self.model_optimizer_name} (momentum = {self.model_optimizer_momentum})")
        print (f"learning rate: {self.model_optimizer_lr}")
        print (f"epoch: {self.train_model_epoch}")
        print (f"batch size: {self.train_model_batch_size}\n")
        # print (f"train loss: {self.train_loss_log}")
        # print (f"validation accuracy (top 1): {self.val_accuracy_log}")
        # print (f"validation accuracy (top 5): {self.val_accuracy_5_log}")
            
        fig, plot = plt.subplots (figsize = (10, 8))
        
        plt.title (f"Train Losses and Validation Accuracies (top1 & top5) for different models")
        plt.yticks([]) 
        legends = []
        
        max_train_loss = 0
        for i in self.train_loss_log: 
            if (max(i) > max_train_loss): max_train_loss = max(i)
        
        
        for i in range (len(self.models)):
            
            exec(f"plot{3*i} = plot.twinx()")
            exec(f"plot{3*i}.set_xlim (0, {self.train_model_epoch} + 1)")
            exec(f"plot{3*i}.set_xlabel (\"Epoch\")")
            exec(f"plot{3*i}.set_ylim (0, max_train_loss*1.2)")
            exec(f"plot{3*i}.set_ylabel (\"Train Loss\")")
            exec(f"plot{3*i}_x = list(range(1, self.train_model_epoch + 1))")
            exec(f"plot{3*i}_y = self.train_loss_log[i]")
            exec(f"plot{3*i}_legend, = plot{3*i}.plot(plot{3*i}_x, plot{3*i}_y, color = {eval(color_schema_1)[i]}, label = \"{self.save_model_name[i]} train.loss\")")
            legends.append(eval(f"plot{3*i}_legend"))
            
            exec(f"plot{3*i+1} = plot.twinx()")
            exec(f"plot{3*i+1}.spines['right'].set_position(('outward', 60))")
            exec(f"plot{3*i+1}.set_xlim (0, {self.train_model_epoch} + 1)")
            exec(f"plot{3*i+1}.set_xlabel (\"Epoch\")")
            exec(f"plot{3*i+1}.set_ylim (0, 1)")
            exec(f"plot{3*i+1}.set_ylabel (\"Validation Accuracy (top 1)\")")
            exec(f"plot{3*i+1}_x = list(range(1, self.train_model_epoch + 1))")
            exec(f"plot{3*i+1}_y = self.val_accuracy_log[i]")
            exec(f"plot{3*i+1}_legend, = plot{3*i+1}.plot(plot{3*i+1}_x, plot{3*i+1}_y, color = {eval(color_schema_2)[i]}, label = \"{self.save_model_name[i]} val.acc1\")")
            legends.append(eval(f"plot{3*i+1}_legend"))
            
            exec(f"plot{3*i+2} = plot.twinx()")
            exec(f"plot{3*i+2}.spines['right'].set_position(('outward', 120))")
            exec(f"plot{3*i+2}.set_xlim (0, {self.train_model_epoch} + 1)")
            exec(f"plot{3*i+2}.set_xlabel (\"Epoch\")")
            exec(f"plot{3*i+2}.set_ylim (0, 1)")
            exec(f"plot{3*i+2}.set_ylabel (\"Validation Accuracy (top 5)\")")
            exec(f"plot{3*i+2}_x = list(range(1, self.train_model_epoch + 1))")
            exec(f"plot{3*i+2}_y = self.val_accuracy_5_log[i]")
            exec(f"plot{3*i+2}_legend, = plot{3*i+2}.plot(plot{3*i+2}_x, plot{3*i+2}_y, color = {eval(color_schema_3)[i]}, label = \"{self.save_model_name[i]} val.acc5\")")
            legends.append(eval(f"plot{3*i+2}_legend"))
                        
            
        plot.legend(handles=legends, bbox_to_anchor=(-0.03, 1))
        
        i = 1; 
        while (os.path.exists(f"{setup.path}/figures/[Experiment] model_trainlossvalacc1valacc5 ({i}).pdf")): i += 1
        plt.savefig(f"{setup.path}/figures/[Experiment] model_trainlossvalacc1valacc5 ({i}).pdf", format="pdf", bbox_inches="tight")
        print (f"\n[Figure] The below figure can be found in \"figures/[Experiment] model_trainlossvalacc1valacc5 ({i}).pdf\"")
        
    
    
    
    
    
    
    def example(self, setup, color_schema_1, color_schema_2):
        
        self.model_optimizer_lr = []
        self.train_model_batch_size = []
        
        self.val_accuracy_log = []
        self.val_accuracy_5_log = []
        
        warning = False
        
        print ("[Experiment] Find the relationship between Models and Train Losses & Validation Accuracies (top 1 & top 5). \n")
        
        for i in range (len(self.models)): 
            
            if (self.models[i].model_optimizer_lr in self.model_optimizer_lr): 
                print (f"Warning: {self.models[i].save_model_name}'s learning rate ({self.models[i].model_optimizer_lr}) already exists! ")
                warning = True
            self.model_optimizer_lr.append(self.models[i].model_optimizer_lr)
            
            if (self.models[i].train_model_batch_size in self.train_model_batch_size): 
                print (f"Warning: {self.models[i].save_model_name}'s batch size ({self.models[i].train_model_batch_size}) already exists! ")
                warning = True
            self.train_model_batch_size.append(self.models[i].train_model_batch_size)
            
            self.val_accuracy_log.append(self.models[i].val_accuracy_log)
            self.val_accuracy_5_log.append(self.models[i].val_accuracy_5_log)
            
            if (i == 0): 
                self.model_name = self.models[i].model_name
                self.dataset = self.models[i].dataset
                self.model_loss_function_name = self.models[i].model_loss_function_name
                self.model_optimizer_name = self.models[i].model_optimizer_name
                self.model_optimizer_lr = self.models[i].model_optimizer_lr
                self.model_optimizer_momentum = self.models[i].model_optimizer_momentum
                self.train_model_epoch = self.models[i].train_model_epoch
                self.train_model_batch_size = self.models[i].train_model_batch_size
            else:
                if (self.models[i].model_name != self.model_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s model ({self.models[i].model_name}) is different from  {self.models[0].save_model_name}'s model ({self.model})! ")
                    warning = True
                if (self.models[i].dataset != self.dataset): 
                    print (f"Warning: {self.models[i].save_model_name}'s dataset ({self.models[i].dataset}) is different from  {self.models[0].save_model_name}'s dataset ({self.dataset})! ")
                    warning = True
                if (self.models[i].model_loss_function_name != self.model_loss_function_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s loss function ({self.models[i].model_loss_function_name}) is different from  {self.models[0].save_model_name}'s loss function ({self.model_loss_function_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_name != self.model_optimizer_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer ({self.models[i].model_optimizer_name}) is different from  {self.models[0].save_model_name}'s optimizer ({self.model_optimizer_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_lr != self.model_optimizer_lr): 
                    print (f"Warning: {self.models[i].save_model_name}'s learning rate ({self.models[i].model_optimizer_lr}) is different from  {self.models[0].save_model_name}'s learning rate ({self.model_optimizer_lr})! ")
                    warning = True
                if (self.models[i].model_optimizer_momentum != self.model_optimizer_momentum): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer momentum ({self.models[i].model_optimizer_momentum}) is different from  {self.models[0].save_model_name}'s optimizer momentum ({self.model_optimizer_momentum})! ")
                    warning = True
                if (self.models[i].train_model_epoch != self.train_model_epoch): 
                    print (f"Warning: {self.models[i].save_model_name}'s train epoches ({self.models[i].train_model_epoch}) is different from  {self.models[0].save_model_name}'s train epoches ({self.train_model_epoch})! ")
                    warning = True
                if (self.models[i].train_model_batch_size != self.train_model_batch_size): 
                    print (f"Warning: {self.models[i].save_model_name}'s batch size ({self.models[i].train_model_batch_size}) is different from  {self.models[0].save_model_name}'s batch size ({self.train_model_batch_size})! ")
                    warning = True

        if (warning): 
            i = input("\nConsider the above warning(s), press Enter to discard or enter anything to continue drawing the chart. ")
            if (i == ""): 
                print ("User has quitted drawing the chart. ")
                return; 
        
        print ("[Controlled Variables]")
        print (f"model: {self.model_name}")
        print (f"dataset: {self.dataset}")
        print (f"loss function: {self.model_loss_function_name}")
        print (f"optimizer: {self.model_optimizer_name} (momentum = {self.model_optimizer_momentum})")
        print (f"learning rate: {self.model_optimizer_lr}")
        print (f"epoch: {self.train_model_epoch}")
        print (f"batch size: {self.train_model_batch_size}\n")
        # print (f"validation accuracy (top 1): {self.val_accuracy_log}")
        # print (f"validation accuracy (top 5): {self.val_accuracy_5_log}")
            
        fig, plot = plt.subplots (figsize = (10, 8))
        
        plt.title (f"Learning Rates, Batch Sizes and Validation Accuracies (top1), Validation Accuracies (top5) for the same model")
        plt.yticks([]) 
        legends = []
        
        for i in range (len(self.models)):
            
            exec(f"plot{3*i} = plot.twinx()")
            exec(f"plot{3*i}.set_xlim (0, {self.train_model_epoch} + 1)")
            exec(f"plot{3*i}.set_xlabel (\"Epoch\")")
            exec(f"plot{3*i}.set_ylim (0, 1)")
            exec(f"plot{3*i}.set_ylabel (\"Validation Accuracy (top 1)\")")
            exec(f"plot{3*i}_x = list(range(1, self.train_model_epoch + 1))")
            exec(f"plot{3*i}_y = self.val_accuracy_log[i]")
            exec(f"plot{3*i}_legend, = plot{3*i}.plot(plot{3*i}_x, plot{3*i}_y, color = {eval(color_schema_1)[i]}, label = \"{self.save_model_name[i]} val.acc1\")")
            legends.append(eval(f"plot{3*i}_legend"))
            
            exec(f"plot{3*i+1} = plot.twinx()")
            exec(f"plot{3*i+1}.set_xlim (0, {self.train_model_epoch} + 1)")
            exec(f"plot{3*i+1}.set_xlabel (\"Epoch\")")
            exec(f"plot{3*i+1}.set_ylim (0, 1)")
            exec(f"plot{3*i+1}.set_ylabel (\"Validation Accuracy (top 5)\")")
            exec(f"plot{3*i+1}_x = list(range(1, self.train_model_epoch + 1))")
            exec(f"plot{3*i+1}_y = self.val_accuracy_5_log[i]")
            exec(f"plot{3*i+1}_legend, = plot{3*i+1}.plot(plot{3*i+1}_x, plot{3*i+1}_y, color = {eval(color_schema_2)[i]}, label = \"{self.save_model_name[i]} val.acc5\")")
            legends.append(eval(f"plot{3*i+1}_legend"))
                        
        plot.legend(handles=legends, bbox_to_anchor=(-0.03, 1))
        
        i = 1; 
        while (os.path.exists(f"{setup.path}/figures/[Experiment] model_example ({i}).pdf")): i += 1
        plt.savefig(f"{setup.path}/figures/[Experiment] model_example ({i}).pdf", format="pdf", bbox_inches="tight")
        print (f"\n[Figure] The below figure can be found in \"figures/[Experiment] model_example ({i}).pdf\"")
    
    
    
    
    
    
    
    
    
    

    
    def lr_valacc1(self, setup, color_schema_1): 
        
        self.model_optimizer_lr = []
        
        self.val_accuracy_log = []
        
        warning = False
        
        print ("[Experiment] Find the relationship between Learning Rates & Validation Accuracies (top 1). \n")
        
        for i in range (len(self.models)): 
            
            if (str(self.models[i].model_optimizer_lr) in self.model_optimizer_lr): 
                print (f"Warning: {self.models[i].save_model_name}'s learning rate ({self.models[i].model_optimizer_lr}) already exists! ")
                warning = True
            self.model_optimizer_lr.append(str(self.models[i].model_optimizer_lr))
            
            self.val_accuracy_log.append(self.models[i].val_accuracy_log)
            
            if (i == 0): 
                self.model_name = self.models[i].model_name
                self.dataset = self.models[i].dataset
                self.model_loss_function_name = self.models[i].model_loss_function_name
                self.model_optimizer_name = self.models[i].model_optimizer_name
                self.model_optimizer_momentum = self.models[i].model_optimizer_momentum
                self.train_model_epoch = self.models[i].train_model_epoch
                self.train_model_batch_size = self.models[i].train_model_batch_size
            else:
                if (self.models[i].model_name != self.model_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s model ({self.models[i].model_name}) is different from  {self.models[0].save_model_name}'s model ({self.model})! ")
                    warning = True
                if (self.models[i].dataset != self.dataset): 
                    print (f"Warning: {self.models[i].save_model_name}'s dataset ({self.models[i].dataset}) is different from  {self.models[0].save_model_name}'s dataset ({self.dataset})! ")
                    warning = True
                if (self.models[i].model_loss_function_name != self.model_loss_function_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s loss function ({self.models[i].model_loss_function_name}) is different from  {self.models[0].save_model_name}'s loss function ({self.model_loss_function_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_name != self.model_optimizer_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer ({self.models[i].model_optimizer_name}) is different from  {self.models[0].save_model_name}'s optimizer ({self.model_optimizer_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_momentum != self.model_optimizer_momentum): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer momentum ({self.models[i].model_optimizer_momentum}) is different from  {self.models[0].save_model_name}'s optimizer momentum ({self.model_optimizer_momentum})! ")
                    warning = True
                if (self.models[i].train_model_epoch != self.train_model_epoch): 
                    print (f"Warning: {self.models[i].save_model_name}'s train epoches ({self.models[i].train_model_epoch}) is different from  {self.models[0].save_model_name}'s train epoches ({self.train_model_epoch})! ")
                    warning = True
                if (self.models[i].train_model_batch_size != self.train_model_batch_size): 
                    print (f"Warning: {self.models[i].save_model_name}'s batch size ({self.models[i].train_model_batch_size}) is different from  {self.models[0].save_model_name}'s batch size ({self.train_model_batch_size})! ")
                    warning = True

        if (warning): 
            i = input("\nConsider the above warning(s), press Enter to discard or enter anything to continue drawing the chart. ")
            if (i == ""): 
                print ("User has quitted drawing the chart. ")
                return; 
        
        print ("[Controlled Variables]")
        print (f"model: {self.model_name}")
        print (f"dataset: {self.dataset}")
        print (f"loss function: {self.model_loss_function_name}")
        print (f"optimizer: {self.model_optimizer_name} (momentum = {self.model_optimizer_momentum})")
        print (f"epoch: {self.train_model_epoch}")
        print (f"batch size: {self.train_model_batch_size}\n")
        print (f"validation accuracy (top 1): {self.val_accuracy_log}")
            
        fig, plot = plt.subplots (figsize = (10, 8))
        
        plt.title (f"Learning Rates and Validation Accuracies (top1) for the same model")
        plt.yticks([]) 
        legends = []
        
        for i in range (len(self.models)):
            
            exec(f"plot{3*i} = plot.twinx()")
            exec(f"plot{3*i}.set_xlim (0, {self.train_model_epoch} + 1)")
            exec(f"plot{3*i}.set_xlabel (\"Epoch\")")
            exec(f"plot{3*i}.set_ylim (0, 1)")
            exec(f"plot{3*i}.set_ylabel (\"Validation Accuracy (top 1)\")")
            exec(f"plot{3*i}_x = list(range(1, self.train_model_epoch + 1))")
            exec(f"plot{3*i}_y = self.val_accuracy_log[i]")
            exec(f"plot{3*i}_legend, = plot{3*i}.plot(plot{3*i}_x, plot{3*i}_y, color = {eval(color_schema_1)[i]}, label = \"lr={self.model_optimizer_lr[i]}\")")
            legends.append(eval(f"plot{3*i}_legend"))
                        
        plot.legend(handles=legends, bbox_to_anchor=(-0.03, 1))
        
        i = 1; 
        while (os.path.exists(f"{setup.path}/figures/[Experiment] lr_valacc1 ({i}).pdf")): i += 1
        plt.savefig(f"{setup.path}/figures/[Experiment] lr_valacc1 ({i}).pdf", format="pdf", bbox_inches="tight")
        print (f"\n[Figure] The below figure can be found in \"figures/[Experiment] lr_valacc1 ({i}).pdf\"")
        
        
        

        
        
    def lr_valacc5(self, setup, color_schema_1): 
        
        self.model_optimizer_lr = []
        
        self.val_accuracy_5_log = []
        
        warning = False
        
        print ("[Experiment] Find the relationship between Learning Rates & Validation Accuracies (top 5). \n")
        
        for i in range (len(self.models)): 
            
            if (str(self.models[i].model_optimizer_lr) in self.model_optimizer_lr): 
                print (f"Warning: {self.models[i].save_model_name}'s learning rate ({self.models[i].model_optimizer_lr}) already exists! ")
                warning = True
            self.model_optimizer_lr.append(str(self.models[i].model_optimizer_lr))
            
            self.val_accuracy_5_log.append(self.models[i].val_accuracy_5_log)
            
            if (i == 0): 
                self.model_name = self.models[i].model_name
                self.dataset = self.models[i].dataset
                self.model_loss_function_name = self.models[i].model_loss_function_name
                self.model_optimizer_name = self.models[i].model_optimizer_name
                self.model_optimizer_momentum = self.models[i].model_optimizer_momentum
                self.train_model_epoch = self.models[i].train_model_epoch
                self.train_model_batch_size = self.models[i].train_model_batch_size
            else:
                if (self.models[i].model_name != self.model_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s model ({self.models[i].model_name}) is different from  {self.models[0].save_model_name}'s model ({self.model})! ")
                    warning = True
                if (self.models[i].dataset != self.dataset): 
                    print (f"Warning: {self.models[i].save_model_name}'s dataset ({self.models[i].dataset}) is different from  {self.models[0].save_model_name}'s dataset ({self.dataset})! ")
                    warning = True
                if (self.models[i].model_loss_function_name != self.model_loss_function_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s loss function ({self.models[i].model_loss_function_name}) is different from  {self.models[0].save_model_name}'s loss function ({self.model_loss_function_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_name != self.model_optimizer_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer ({self.models[i].model_optimizer_name}) is different from  {self.models[0].save_model_name}'s optimizer ({self.model_optimizer_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_momentum != self.model_optimizer_momentum): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer momentum ({self.models[i].model_optimizer_momentum}) is different from  {self.models[0].save_model_name}'s optimizer momentum ({self.model_optimizer_momentum})! ")
                    warning = True
                if (self.models[i].train_model_epoch != self.train_model_epoch): 
                    print (f"Warning: {self.models[i].save_model_name}'s train epoches ({self.models[i].train_model_epoch}) is different from  {self.models[0].save_model_name}'s train epoches ({self.train_model_epoch})! ")
                    warning = True
                if (self.models[i].train_model_batch_size != self.train_model_batch_size): 
                    print (f"Warning: {self.models[i].save_model_name}'s batch size ({self.models[i].train_model_batch_size}) is different from  {self.models[0].save_model_name}'s batch size ({self.train_model_batch_size})! ")
                    warning = True

        if (warning): 
            i = input("\nConsider the above warning(s), press Enter to discard or enter anything to continue drawing the chart. ")
            if (i == ""): 
                print ("User has quitted drawing the chart. ")
                return; 
        
        print ("[Controlled Variables]")
        print (f"model: {self.model_name}")
        print (f"dataset: {self.dataset}")
        print (f"loss function: {self.model_loss_function_name}")
        print (f"optimizer: {self.model_optimizer_name} (momentum = {self.model_optimizer_momentum})")
        print (f"epoch: {self.train_model_epoch}")
        print (f"batch size: {self.train_model_batch_size}\n")
        print (f"validation accuracy (top 5): {self.val_accuracy_5_log}")
            
        fig, plot = plt.subplots (figsize = (10, 8))
        
        plt.title (f"Learning Rates and Validation Accuracies (top1) for the same model")
        plt.yticks([]) 
        legends = []
        
        for i in range (len(self.models)):
            
            exec(f"plot{3*i} = plot.twinx()")
            exec(f"plot{3*i}.set_xlim (0, {self.train_model_epoch} + 1)")
            exec(f"plot{3*i}.set_xlabel (\"Epoch\")")
            exec(f"plot{3*i}.set_ylim (0, 1)")
            exec(f"plot{3*i}.set_ylabel (\"Validation Accuracy (top 1)\")")
            exec(f"plot{3*i}_x = list(range(1, self.train_model_epoch + 1))")
            exec(f"plot{3*i}_y = self.val_accuracy_5_log[i]")
            exec(f"plot{3*i}_legend, = plot{3*i}.plot(plot{3*i}_x, plot{3*i}_y, color = {eval(color_schema_1)[i]}, label = \"lr={self.model_optimizer_lr[i]}\")")
            legends.append(eval(f"plot{3*i}_legend"))
                        
        plot.legend(handles=legends, bbox_to_anchor=(-0.03, 1))
        
        i = 1; 
        while (os.path.exists(f"{setup.path}/figures/[Experiment] lr_valacc5 ({i}).pdf")): i += 1
        plt.savefig(f"{setup.path}/figures/[Experiment] lr_valacc5 ({i}).pdf", format="pdf", bbox_inches="tight")
        print (f"\n[Figure] The below figure can be found in \"figures/[Experiment] lr_valacc5 ({i}).pdf\"")
        
        
        
        
    
    def bs_valacc1(self, setup, color_schema):
        
        self.train_model_batch_size = []
        self.val_accuracy_log = []
        
        print ("[Experiment] Find the relationship between Batch Sizes and Validation Accuracies. \n")
        warning = False
        for i in range (len(self.models)):
            
            # self.train_model_batch_size = self.models[i].train_model_batch_size
            if (self.models[i].train_model_batch_size in self.train_model_batch_size): 
                print (f"Warning: {self.models[i].save_model_name}'s batch size ({self.models[i].train_model_batch_size}) already exists! ")
                warning = True
            self.val_accuracy_log.append(self.models[i].val_accuracy_log)
                
            if (i == 0): 
                self.model_name = self.models[i].model_name
                self.dataset = self.models[i].dataset
                self.model_loss_function_name = self.models[i].model_loss_function_name
                self.model_optimizer_name = self.models[i].model_optimizer_name
                self.model_optimizer_lr = self.models[i].model_optimizer_lr
                self.model_optimizer_momentum = self.models[i].model_optimizer_momentum
                self.train_model_epoch = self.models[i].train_model_epoch
                self.train_model_batch_size.append(self.models[i].train_model_batch_size)
                
            else:
                if (self.models[i].model_name != self.model_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s model ({self.models[i].model_name}) is different from  {self.models[0].save_model_name}'s model ({self.model_name})! ")
                    warning = True
                if (self.models[i].dataset != self.dataset): 
                    print (f"Warning: {self.models[i].save_model_name}'s dataset ({self.models[i].dataset}) is different from  {self.models[0].save_model_name}'s dataset ({self.dataset})! ")
                    warning = True
                if (self.models[i].model_loss_function_name != self.model_loss_function_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s loss function ({self.models[i].model_loss_function_name}) is different from  {self.models[0].save_model_name}'s loss function ({self.model_loss_function_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_name != self.model_optimizer_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer ({self.models[i].model_optimizer_name}) is different from  {self.models[0].save_model_name}'s optimizer ({self.model_optimizer_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_lr != self.model_optimizer_lr): 
                    print (f"Warning: {self.models[i].save_model_name}'s learning rate ({self.models[i].model_optimizer_lr}) is different from  {self.models[0].save_model_name}'s learning rate ({self.model_optimizer_lr})! ")
                    warning = True
                if (self.models[i].model_optimizer_momentum != self.model_optimizer_momentum): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer momentum ({self.models[i].model_optimizer_momentum}) is different from  {self.models[0].save_model_name}'s optimizer momentum ({self.model_optimizer_momentum})! ")
                    warning = True
                if (self.models[i].train_model_epoch != self.train_model_epoch): 
                    print (f"Warning: {self.models[i].save_model_name}'s train epoches ({self.models[i].train_model_epoch}) is different from  {self.models[0].save_model_name}'s train epoches ({self.train_model_epoch})! ")
                    warning = True
                

        if (warning): 
            i = input("\nConsider the above warning(s), press Enter to discard or enter anything to continue drawing the chart. ")
            if (i == ""): 
                print ("User has quitted drawing the chart. ")
                return; 
        
        print ("[Controlled Variables]")
        print (f"model: {self.model_name}")
        print (f"dataset: {self.dataset}")
        print (f"loss function: {self.model_loss_function_name}")
        print (f"optimizer: {self.model_optimizer_name} (momentum = {self.model_optimizer_momentum})")
        print (f"learning rate: {self.model_optimizer_lr}")
        print (f"epoch: {self.train_model_epoch}")
            
        fig, plot = plt.subplots (figsize = (10, 8))
        plt.title (f"Validation Accuracy for {self.models[0].model_name} models (different bs)")
        # x-axis: epoch
        # y-axis: val accuracy
        legends = []
        
        for i in range (len(self.models)):
            exec(f"plot{i} = plot.twinx()")
            exec(f"plot{i}.set_xlim (0, {self.models[0].train_model_epoch} + 1)")
            exec(f"plot{i}.set_xlabel (\"Epoch\")")
            exec(f"plot{i}.set_ylim (0, 1.2)")
            exec(f"plot{i}.set_ylabel (\"Validation Accuracy\")")
            exec(f"plot{i}_x = list(range(1, len(self.models[i].train_loss_log) + 1))")
            exec(f"plot{i}_y = self.models[i].val_accuracy_log")
            exec(f"plot{i}_legend, = plot{i}.plot(plot{i}_x, plot{i}_y, color = {eval(color_schema)[i]}, label = \"bs={self.models[i].train_model_batch_size}\")")
            legends.append(eval(f"plot{i}_legend"))
        plot.legend(handles=legends, loc=1)
        
        i = 1; 
        while (os.path.exists(f"{setup.path}/figures/[Experiment] bs_valacc1 ({i}).pdf")): i += 1
        plt.savefig(f"{setup.path}/figures/[Experiment] bs_valacc1 ({i}).pdf", format="pdf", bbox_inches="tight")
        print (f"\n[Figure] The below figure can be found in \"figures/[Experiment] bs_valacc1 ({i}).pdf\"")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    def model_lr_valacc1(self, setup, color_schema): 

        self.model_name = []
        self.model_optimizer_lr = []
        self.val_accuracy_log = []
        
        print ("[Experiment] Find the relationship between Model & Learning Rates and Val Accuracy. \n")
        warning = False
        
        for i in range (len(self.models)): 

            # self.model_name = self.models[i].model_name
            if (str(self.models[i].model_name) not in self.model_name): 
                self.model_name.append(str(self.models[i].model_name))            
            
            # self.model_optimizer_lr = self.models[i].model_optimizer_lr
            if (self.models[i].model_optimizer_lr not in self.model_optimizer_lr): 
                self.model_optimizer_lr.append(self.models[i].model_optimizer_lr)
            
            self.val_accuracy_log.append(self.models[i].val_accuracy_log)
            
            if (i == 0): 
                self.dataset = self.models[i].dataset
                self.model_loss_function_name = self.models[i].model_loss_function_name
                self.model_optimizer_name = self.models[i].model_optimizer_name
                self.model_optimizer_momentum = self.models[i].model_optimizer_momentum
                self.train_model_epoch = self.models[i].train_model_epoch
                self.train_model_batch_size = self.models[i].train_model_batch_size
            else:
                if (self.models[i].dataset != self.dataset): 
                    print (f"Warning: {self.models[i].save_model_name}'s dataset ({self.models[i].dataset}) is different from  {self.models[0].save_model_name}'s dataset ({self.dataset})! ")
                    warning = True
                if (self.models[i].model_loss_function_name != self.model_loss_function_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s loss function ({self.models[i].model_loss_function_name}) is different from  {self.models[0].save_model_name}'s loss function ({self.model_loss_function_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_name != self.model_optimizer_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer ({self.models[i].model_optimizer_name}) is different from  {self.models[0].save_model_name}'s optimizer ({self.model_optimizer_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_momentum != self.model_optimizer_momentum): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer momentum ({self.models[i].model_optimizer_momentum}) is different from  {self.models[0].save_model_name}'s optimizer momentum ({self.model_optimizer_momentum})! ")
                    warning = True
                if (self.models[i].train_model_epoch != self.train_model_epoch): 
                    print (f"Warning: {self.models[i].save_model_name}'s train epoches ({self.models[i].train_model_epoch}) is different from  {self.models[0].save_model_name}'s train epoches ({self.train_model_epoch})! ")
                    warning = True
                if (self.models[i].train_model_batch_size != self.train_model_batch_size): 
                    print (f"Warning: {self.models[i].save_model_name}'s batch size ({self.models[i].train_model_batch_size}) is different from  {self.models[0].save_model_name}'s batch size ({self.train_model_batch_size})! ")
                    warning = True
                    
        if (warning): 
            i = input("\nConsider the above warning(s), press Enter to discard or enter anything to continue drawing the chart. ")
            if (i == ""): 
                print ("User has quitted drawing the chart. ")
                return; 
        
        print ("[Controlled Variables]")
        print (f"dataset: {self.dataset}")
        print (f"loss function: {self.model_loss_function_name}")
        print (f"optimizer: {self.model_optimizer_name} (momentum = {self.model_optimizer_momentum})")
        print (f"epoch: {self.train_model_epoch}")
        print (f"batch size: {self.train_model_batch_size}\n")
        print (f"model: {self.model_name}")
        print (f"learning rate: {self.model_optimizer_lr}")
        val_accuracy_log = [i[-1] for i in self.val_accuracy_log]
        print (f"val accuracy: {val_accuracy_log}")
        
        
        fig, plot = plt.subplots (figsize = (10, 8))
        plt.title (f"Validation Accuracy for {self.models[0].model_name} models (different lr)")
        # x-axis: learning rate
        # y-axis: train_time       
        
        legends = []
        
        plot_x = sorted(self.model_optimizer_lr)
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.xticks(rotation=90)
        for i in range (len(plot_x)): 
            plot_x[i] = str(plot_x[i])
        exec(f"plt.xticks(range(0, len(plot_x)), plot_x, rotation='vertical')")
        exec(f"plt.margins(0.15)") 
        
        
        for a in range (len(self.model_name)): 
            
            import collections
            dictionary = []
            for i in self.models: 
                if (i.model_name == self.model_name[a]): 
                    dictionary.append((i.model_optimizer_lr, i.val_accuracy_log[-1]))
            dictionary = sorted(dictionary, key=lambda d:d[0])
            print (dictionary)
            
            exec(f"plot{a} = plot.twinx()")
            exec(f"plot{a}.set_xlabel (\"Learning Rate\")")
            exec(f"plot{a}.set_ylim (0, 1.2)")
            exec(f"plot{a}.set_ylabel (\"Training Time\")")
            exec(f"plot{a}_x = []")
            exec(f"plot{a}_y = []")
            for x, y in dictionary: 
                exec(f"plot{a}_x.append(str(x))")
                exec(f"plot{a}_y.append(y)")
            exec(f"plot{a}_legend, = plot{a}.plot(plot{a}_x, plot{a}_y, color={eval(color_schema)[a]}, label = \"{self.model_name[a]}\")")
            legends.append(eval(f"plot{a}_legend"))

        plot.legend(handles=legends, loc=1)
        i = 1; 
        while (os.path.exists(f"{setup.path}/figures/[Experiment] model_lr_valacc1 ({i}).pdf")): i += 1
        plt.savefig(f"{setup.path}/figures/[Experiment] model_lr_valacc1 ({i}).pdf", format="pdf", bbox_inches="tight")
        print (f"\n[Figure] The below figure can be found in \"figures/[Experiment] model_lr_valacc1 ({i}).pdf\"")  

        

        
    def model_lr_time(self, setup, color_schema):

        self.model_name = []
        self.model_optimizer_lr = []
        self.train_time_epoch = []
        
        print ("[Experiment] Find the relationship between Model & Learning Rates and Train Time (per epoch). \n")
        warning = False
        
        for i in range (len(self.models)): 

            # self.model_name = self.models[i].model_name
            if (str(self.models[i].model_name) not in self.model_name): 
                self.model_name.append(str(self.models[i].model_name))            
            
            # self.model_optimizer_lr = self.models[i].model_optimizer_lr
            if (self.models[i].model_optimizer_lr not in self.model_optimizer_lr): 
                self.model_optimizer_lr.append(self.models[i].model_optimizer_lr)
            
            self.train_time_epoch.append(self.models[i].train_time_epoch)
            
            if (i == 0): 
                self.dataset = self.models[i].dataset
                self.model_loss_function_name = self.models[i].model_loss_function_name
                self.model_optimizer_name = self.models[i].model_optimizer_name
                self.model_optimizer_momentum = self.models[i].model_optimizer_momentum
                self.train_model_epoch = self.models[i].train_model_epoch
                self.train_model_batch_size = self.models[i].train_model_batch_size
            else:
                if (self.models[i].dataset != self.dataset): 
                    print (f"Warning: {self.models[i].save_model_name}'s dataset ({self.models[i].dataset}) is different from  {self.models[0].save_model_name}'s dataset ({self.dataset})! ")
                    warning = True
                if (self.models[i].model_loss_function_name != self.model_loss_function_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s loss function ({self.models[i].model_loss_function_name}) is different from  {self.models[0].save_model_name}'s loss function ({self.model_loss_function_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_name != self.model_optimizer_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer ({self.models[i].model_optimizer_name}) is different from  {self.models[0].save_model_name}'s optimizer ({self.model_optimizer_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_momentum != self.model_optimizer_momentum): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer momentum ({self.models[i].model_optimizer_momentum}) is different from  {self.models[0].save_model_name}'s optimizer momentum ({self.model_optimizer_momentum})! ")
                    warning = True
                if (self.models[i].train_model_epoch != self.train_model_epoch): 
                    print (f"Warning: {self.models[i].save_model_name}'s train epoches ({self.models[i].train_model_epoch}) is different from  {self.models[0].save_model_name}'s train epoches ({self.train_model_epoch})! ")
                    warning = True
                if (self.models[i].train_model_batch_size != self.train_model_batch_size): 
                    print (f"Warning: {self.models[i].save_model_name}'s batch size ({self.models[i].train_model_batch_size}) is different from  {self.models[0].save_model_name}'s batch size ({self.train_model_batch_size})! ")
                    warning = True
                    
        if (warning): 
            i = input("\nConsider the above warning(s), press Enter to discard or enter anything to continue drawing the chart. ")
            if (i == ""): 
                print ("User has quitted drawing the chart. ")
                return; 
        
        print ("[Controlled Variables]")
        print (f"dataset: {self.dataset}")
        print (f"loss function: {self.model_loss_function_name}")
        print (f"optimizer: {self.model_optimizer_name} (momentum = {self.model_optimizer_momentum})")
        print (f"epoch: {self.train_model_epoch}")
        print (f"batch size: {self.train_model_batch_size}\n")
        print (f"model: {self.model_name}")
        print (f"learning rate: {self.model_optimizer_lr}")
        print (f"time (per epoch): {self.train_time_epoch}")
        
        
        fig, plot = plt.subplots (figsize = (10, 8))
        plt.title (f"Train Time (per epoch) for {self.models[0].model_name} models (different lr)")
        # x-axis: learning rate
        # y-axis: train_time       
        
        legends = []
        
        plot_x = sorted(self.model_optimizer_lr)
        plot_y = max(self.train_time_epoch)
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.xticks(rotation=90)
        for i in range (len(plot_x)): 
            plot_x[i] = str(plot_x[i])
        exec(f"plt.xticks(range(0, len(plot_x)), plot_x, rotation='vertical')")
        exec(f"plt.margins(0.15)") 
        
        
        for a in range (len(self.model_name)): 
            
            import collections
            dictionary = []
            for i in self.models: 
                if (i.model_name == self.model_name[a]): 
                    dictionary.append((i.model_optimizer_lr, i.train_time_epoch))
            dictionary = sorted(dictionary, key=lambda d:d[0])
            print (dictionary)
            
            exec(f"plot{a} = plot.twinx()")
            exec(f"plot{a}.set_xlabel (\"Learning Rate\")")
            exec(f"plot{a}.set_ylim (0, {plot_y * 1.5})")
            exec(f"plot{a}.set_ylabel (\"Training Time\")")
            exec(f"plot{a}_x = []")
            exec(f"plot{a}_y = []")
            for x, y in dictionary: 
                exec(f"plot{a}_x.append(str(x))")
                exec(f"plot{a}_y.append(y)")
            exec(f"plot{a}_legend, = plot{a}.plot(plot{a}_x, plot{a}_y, color={eval(color_schema)[a]}, label = \"{self.model_name[a]}\")")
            legends.append(eval(f"plot{a}_legend"))

        plot.legend(handles=legends, loc=1)
        i = 1; 
        while (os.path.exists(f"{setup.path}/figures/[Experiment] model_lr_time ({i}).pdf")): i += 1
        plt.savefig(f"{setup.path}/figures/[Experiment] model_lr_time ({i}).pdf", format="pdf", bbox_inches="tight")
        print (f"\n[Figure] The below figure can be found in \"figures/[Experiment] model_lr_time ({i}).pdf\"")
        
        
        
        
    def model_bs_valacc1(self, setup, color_schema):
        
        self.model_name = []
        self.train_model_batch_size = []
        self.val_accuracy_log = []
        
        print ("[Experiment] Find the relationship between Model & Learning Rates and Val Accuracy. \n")
        warning = False
        
        for i in range (len(self.models)): 

            # self.model_name = self.models[i].model_name
            if (str(self.models[i].model_name) not in self.model_name): 
                self.model_name.append(str(self.models[i].model_name))            
            
            # self.model_optimizer_lr = self.models[i].model_optimizer_lr
            if (self.models[i].train_model_batch_size not in self.train_model_batch_size): 
                self.train_model_batch_size.append(self.models[i].train_model_batch_size)
            
            self.val_accuracy_log.append(self.models[i].val_accuracy_log)
            
            if (i == 0): 
                self.dataset = self.models[i].dataset
                self.model_loss_function_name = self.models[i].model_loss_function_name
                self.model_optimizer_name = self.models[i].model_optimizer_name
                self.model_optimizer_momentum = self.models[i].model_optimizer_momentum
                self.train_model_epoch = self.models[i].train_model_epoch
                self.model_optimizer_lr = self.models[i].model_optimizer_lr
            else:
                if (self.models[i].dataset != self.dataset): 
                    print (f"Warning: {self.models[i].save_model_name}'s dataset ({self.models[i].dataset}) is different from  {self.models[0].save_model_name}'s dataset ({self.dataset})! ")
                    warning = True
                if (self.models[i].model_loss_function_name != self.model_loss_function_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s loss function ({self.models[i].model_loss_function_name}) is different from  {self.models[0].save_model_name}'s loss function ({self.model_loss_function_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_name != self.model_optimizer_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer ({self.models[i].model_optimizer_name}) is different from  {self.models[0].save_model_name}'s optimizer ({self.model_optimizer_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_lr != self.model_optimizer_lr): 
                    print (f"Warning: {self.models[i].save_model_name}'s learning rate ({self.models[i].model_optimizer_lr}) is different from  {self.models[0].save_model_name}'s learning rate ({self.model_optimizer_lr})! ")
                    warning = True
                if (self.models[i].model_optimizer_momentum != self.model_optimizer_momentum): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer momentum ({self.models[i].model_optimizer_momentum}) is different from  {self.models[0].save_model_name}'s optimizer momentum ({self.model_optimizer_momentum})! ")
                    warning = True
                if (self.models[i].train_model_epoch != self.train_model_epoch): 
                    print (f"Warning: {self.models[i].save_model_name}'s train epoches ({self.models[i].train_model_epoch}) is different from  {self.models[0].save_model_name}'s train epoches ({self.train_model_epoch})! ")
                    warning = True
                    
        if (warning): 
            i = input("\nConsider the above warning(s), press Enter to discard or enter anything to continue drawing the chart. ")
            if (i == ""): 
                print ("User has quitted drawing the chart. ")
                return; 
        
        print ("[Controlled Variables]")
        print (f"dataset: {self.dataset}")
        print (f"loss function: {self.model_loss_function_name}")
        print (f"optimizer: {self.model_optimizer_name} (momentum = {self.model_optimizer_momentum})")
        print (f"learning rate: {self.model_optimizer_lr}")
        print (f"epoch: {self.train_model_epoch}\n")
        print (f"model: {self.model_name}")
        print (f"batch size: {self.train_model_batch_size}")
        val_accuracy_log = [i[-1] for i in self.val_accuracy_log]
        print (f"val accuracy: {val_accuracy_log}")
        
        
        fig, plot = plt.subplots (figsize = (10, 8))
        plt.title (f"Validation Accuracy for {self.models[0].model_name} models (different bs)")
        # x-axis: learning rate
        # y-axis: train_time       
        
        legends = []
        
        plot_x = sorted(self.train_model_batch_size)
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.xticks(rotation=90)
        for i in range (len(plot_x)): 
            plot_x[i] = str(plot_x[i])
        exec(f"plt.xticks(range(0, len(plot_x)), plot_x, rotation='vertical')")
        exec(f"plt.margins(0.15)") 
        
        
        for a in range (len(self.model_name)): 
            
            import collections
            dictionary = []
            for i in self.models: 
                if (i.model_name == self.model_name[a]): 
                    dictionary.append((i.train_model_batch_size, i.val_accuracy_log[-1]))
            dictionary = sorted(dictionary, key=lambda d:d[0])
            print (dictionary)
            
            exec(f"plot{a} = plot.twinx()")
            exec(f"plot{a}.set_xlabel (\"Batch Size\")")
            exec(f"plot{a}.set_ylim (0, 1.2)")
            exec(f"plot{a}.set_ylabel (\"Training Time\")")
            exec(f"plot{a}_x = []")
            exec(f"plot{a}_y = []")
            for x, y in dictionary: 
                exec(f"plot{a}_x.append(str(x))")
                exec(f"plot{a}_y.append(y)")
            exec(f"plot{a}_legend, = plot{a}.plot(plot{a}_x, plot{a}_y, color={eval(color_schema)[a]}, label = \"{self.model_name[a]}\")")
            legends.append(eval(f"plot{a}_legend"))

        plot.legend(handles=legends, loc=1)
        i = 1; 
        while (os.path.exists(f"{setup.path}/figures/[Experiment] model_bs_valacc1 ({i}).pdf")): i += 1
        plt.savefig(f"{setup.path}/figures/[Experiment] model_bs_valacc1 ({i}).pdf", format="pdf", bbox_inches="tight")
        print (f"\n[Figure] The below figure can be found in \"figures/[Experiment] model_bs_valacc1 ({i}).pdf\"")  
    
    
    
    def model_bs_time(self, setup, color_schema):

        self.model_name = []
        self.train_model_batch_size = []
        self.train_time_epoch = []
        
        print ("[Experiment] Find the relationship between Model & Batch Sizes and Train Time (per epoch). \n")
        warning = False
        
        for i in range (len(self.models)): 

            # self.model_name = self.models[i].model_name
            if (str(self.models[i].model_name) not in self.model_name): 
                self.model_name.append(str(self.models[i].model_name))            
            
            # self.train_model_batch_size = self.models[i].train_model_batch_size
            if (self.models[i].train_model_batch_size not in self.train_model_batch_size): 
                self.train_model_batch_size.append(self.models[i].train_model_batch_size)
            
            self.train_time_epoch.append(self.models[i].train_time_epoch)
            
            if (i == 0): 
                self.dataset = self.models[i].dataset
                self.model_loss_function_name = self.models[i].model_loss_function_name
                self.model_optimizer_name = self.models[i].model_optimizer_name
                self.model_optimizer_momentum = self.models[i].model_optimizer_momentum
                self.train_model_epoch = self.models[i].train_model_epoch
                self.model_optimizer_lr = self.models[i].model_optimizer_lr
            else:
                if (self.models[i].dataset != self.dataset): 
                    print (f"Warning: {self.models[i].save_model_name}'s dataset ({self.models[i].dataset}) is different from  {self.models[0].save_model_name}'s dataset ({self.dataset})! ")
                    warning = True
                if (self.models[i].model_loss_function_name != self.model_loss_function_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s loss function ({self.models[i].model_loss_function_name}) is different from  {self.models[0].save_model_name}'s loss function ({self.model_loss_function_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_name != self.model_optimizer_name): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer ({self.models[i].model_optimizer_name}) is different from  {self.models[0].save_model_name}'s optimizer ({self.model_optimizer_name})! ")
                    warning = True
                if (self.models[i].model_optimizer_lr != self.model_optimizer_lr): 
                    print (f"Warning: {self.models[i].save_model_name}'s learning rate ({self.models[i].model_optimizer_lr}) is different from  {self.models[0].save_model_name}'s learning rate ({self.model_optimizer_lr})! ")
                    warning = True
                if (self.models[i].model_optimizer_momentum != self.model_optimizer_momentum): 
                    print (f"Warning: {self.models[i].save_model_name}'s optimizer momentum ({self.models[i].model_optimizer_momentum}) is different from  {self.models[0].save_model_name}'s optimizer momentum ({self.model_optimizer_momentum})! ")
                    warning = True
                if (self.models[i].train_model_epoch != self.train_model_epoch): 
                    print (f"Warning: {self.models[i].save_model_name}'s train epoches ({self.models[i].train_model_epoch}) is different from  {self.models[0].save_model_name}'s train epoches ({self.train_model_epoch})! ")
                    warning = True
                    
        if (warning): 
            i = input("\nConsider the above warning(s), press Enter to discard or enter anything to continue drawing the chart. ")
            if (i == ""): 
                print ("User has quitted drawing the chart. ")
                return; 
        
        print ("[Controlled Variables]")
        print (f"dataset: {self.dataset}")
        print (f"loss function: {self.model_loss_function_name}")
        print (f"optimizer: {self.model_optimizer_name} (momentum = {self.model_optimizer_momentum})")
        print (f"learning rate: {self.model_optimizer_lr}")
        print (f"epoch: {self.train_model_epoch}")
        print (f"model: {self.model_name}")
        print (f"batch size: {self.train_model_batch_size}")
        print (f"time (per epoch): {self.train_time_epoch}")
        
        
        fig, plot = plt.subplots (figsize = (10, 8))
        plt.title (f"Train Time (per epoch) for {self.models[0].model_name} models (different lr)")
        # x-axis: learning rate
        # y-axis: train_time       
        
        legends = []
        
        plot_x = sorted(self.train_model_batch_size)
        plot_y = max(self.train_time_epoch)
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.xticks(rotation=90)
        for i in range (len(plot_x)): 
            plot_x[i] = str(plot_x[i])
        exec(f"plt.xticks(range(0, len(plot_x)), plot_x, rotation='vertical')")
        exec(f"plt.margins(0.15)") 
        
        
        for a in range (len(self.model_name)): 
            
            import collections
            dictionary = []
            for i in self.models: 
                if (i.model_name == self.model_name[a]): 
                    dictionary.append((i.train_model_batch_size, i.train_time_epoch))
            dictionary = sorted(dictionary, key=lambda d:d[0])
            print (dictionary)
            
            exec(f"plot{a} = plot.twinx()")
            exec(f"plot{a}.set_xlabel (\"Learning Rate\")")
            exec(f"plot{a}.set_ylim (0, {plot_y * 1.5})")
            exec(f"plot{a}.set_ylabel (\"Training Time\")")
            exec(f"plot{a}_x = []")
            exec(f"plot{a}_y = []")
            for x, y in dictionary: 
                exec(f"plot{a}_x.append(str(x))")
                exec(f"plot{a}_y.append(y)")
            exec(f"plot{a}_legend, = plot{a}.plot(plot{a}_x, plot{a}_y, color={eval(color_schema)[a]}, label = \"{self.model_name[a]}\")")
            legends.append(eval(f"plot{a}_legend"))

        plot.legend(handles=legends, loc=1)
        i = 1; 
        while (os.path.exists(f"{setup.path}/figures/[Experiment] model_bs_time ({i}).pdf")): i += 1
        plt.savefig(f"{setup.path}/figures/[Experiment] model_bs_time ({i}).pdf", format="pdf", bbox_inches="tight")
        print (f"\n[Figure] The below figure can be found in \"figures/[Experiment] model_bs_time ({i}).pdf\"")
    
    
    
    def model_lr_bs_valacc1(self, setup, color_schema):
        pass
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    def train_initial_confusion(self, setup): 
        
        if (len(self.models) != 1): 
            raise Exception ("[Error] Only one model's confusion matrix is supported at one time. ")
        model = self.models[0]
        classes = ["class_1", "class_2", "class_3", "class_4", "class_5", "class_6", "class_7", "class_8", "class_9", "class_10", "class_11", "class_12", "class_13", "class_14", "class_15", "class_16", "class_17"]
        
        # Referece: Previous work
        # https://gitfront.io/r/user-3667130/cfbb432096a5c845e9aa630e90ece8ab4ef08ff8/comp3340_group10_image-classification/blob/experiment.py
        
        cf_matrix = confusion_matrix (model.train_prediction_class[0].data.cpu().numpy(), model.train_label_class[0].data.cpu().numpy())
        df_cm = pd.DataFrame(
            cf_matrix/cf_matrix.astype(np.float64).sum(axis=1), 
            index=[i for i in classes],
            columns=[i for i in classes]
        )
        plt.figure(figsize=(10, 8))    
        fig = sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt=".2f").get_figure()
        i = 1; 
        while (os.path.exists(f"{setup.path}/figures/[Experiment] train_initial_confusion ({i}).pdf")): i += 1
        fig.savefig(f"{setup.path}/figures/[Experiment] train_initial_confusion ({i}).pdf", format="pdf", bbox_inches="tight")
        print (f"\n[Figure] The below figure can be found in \"figures/[Experiment] train_initial_confusion ({i}).pdf\"")
        

    def train_last_confusion(self, setup): 
        
        if (len(self.models) != 1): 
            raise Exception ("[Error] Only one model's confusion matrix is supported at one time. ")
        model = self.models[0]
        classes = ["class_1", "class_2", "class_3", "class_4", "class_5", "class_6", "class_7", "class_8", "class_9", "class_10", "class_11", "class_12", "class_13", "class_14", "class_15", "class_16", "class_17"]
        
        # Referece: Previous work
        # https://gitfront.io/r/user-3667130/cfbb432096a5c845e9aa630e90ece8ab4ef08ff8/comp3340_group10_image-classification/blob/experiment.py
        
        cf_matrix = confusion_matrix (model.train_prediction_class[-1].data.cpu().numpy(), model.train_label_class[-1].data.cpu().numpy())
        df_cm = pd.DataFrame(
            cf_matrix/cf_matrix.astype(np.float64).sum(axis=1), 
            index=[i for i in classes],
            columns=[i for i in classes]
        )
        plt.figure(figsize=(10, 8))    
        fig = sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt=".2f").get_figure()
        i = 1; 
        while (os.path.exists(f"{setup.path}/figures/[Experiment] train_last_confusion ({i}).pdf")): i += 1
        fig.savefig(f"{setup.path}/figures/[Experiment] train_last_confusion ({i}).pdf", format="pdf", bbox_inches="tight")
        print (f"\n[Figure] The below figure can be found in \"figures/[Experiment] train_last_confusion ({i}).pdf\"")


    def val_initial_confusion(self, setup): 
        
        if (len(self.models) != 1): 
            raise Exception ("[Error] Only one model's confusion matrix is supported at one time. ")
        model = self.models[0]
        classes = ["class_1", "class_2", "class_3", "class_4", "class_5", "class_6", "class_7", "class_8", "class_9", "class_10", "class_11", "class_12", "class_13", "class_14", "class_15", "class_16", "class_17"]
        
        # Referece: Previous work
        # https://gitfront.io/r/user-3667130/cfbb432096a5c845e9aa630e90ece8ab4ef08ff8/comp3340_group10_image-classification/blob/experiment.py
        
        # print (model.val_prediction_class[0].data.cpu().numpy())
        # print (model.val_label_class[0].data.cpu().numpy())
        cf_matrix = confusion_matrix (model.val_prediction_class[0].data.cpu().numpy(), model.val_label_class[0].data.cpu().numpy())
        df_cm = pd.DataFrame(
            cf_matrix/cf_matrix.astype(np.float64).sum(axis=1), 
            index=[i for i in classes],
            columns=[i for i in classes]
        )
        plt.figure(figsize=(10, 8))    
        fig = sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt=".2f").get_figure()
        i = 1; 
        while (os.path.exists(f"{setup.path}/figures/[Experiment] val_initial_confusion ({i}).pdf")): i += 1
        fig.savefig(f"{setup.path}/figures/[Experiment] val_initial_confusion ({i}).pdf", format="pdf", bbox_inches="tight")
        print (f"\n[Figure] The below figure can be found in \"figures/[Experiment] val_initial_confusion ({i}).pdf\"")
        
        
    
    
    def val_last_confusion(self, setup): 
        
        if (len(self.models) != 1): 
            raise Exception ("[Error] Only one model's confusion matrix is supported at one time. ")
        model = self.models[0]
        classes = ["class_1", "class_2", "class_3", "class_4", "class_5", "class_6", "class_7", "class_8", "class_9", "class_10", "class_11", "class_12", "class_13", "class_14", "class_15", "class_16", "class_17"]
        
        # Referece: Previous work
        # https://gitfront.io/r/user-3667130/cfbb432096a5c845e9aa630e90ece8ab4ef08ff8/comp3340_group10_image-classification/blob/experiment.py
        
        cf_matrix = confusion_matrix (model.val_prediction_class[-1].data.cpu().numpy(), model.val_label_class[-1].data.cpu().numpy())
        df_cm = pd.DataFrame(
            cf_matrix/cf_matrix.astype(np.float64).sum(axis=1), 
            index=[i for i in classes],
            columns=[i for i in classes]
        )
        plt.figure(figsize=(10, 8))    
        fig = sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt=".2f").get_figure()
        i = 1; 
        while (os.path.exists(f"{setup.path}/figures/[Experiment] val_last_confusion ({i}).pdf")): i += 1
        fig.savefig(f"{setup.path}/figures/[Experiment] val_last_confusion ({i}).pdf", format="pdf", bbox_inches="tight")
        print (f"\n[Figure] The below figure can be found in \"figures/[Experiment] val_last_confusion ({i}).pdf\"")
        

        