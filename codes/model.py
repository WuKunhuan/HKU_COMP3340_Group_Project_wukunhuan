
import os
from IPython.display import clear_output
import math, shutil, time

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms

class model_schema: 
    
    def __init__(self): 
        self.dataset = None
        self.model_name = None
        self.model_loss_function_name = None
        self.model_optimizer_name = None
        self.model_optimizer_lr = None
        self.model_optimizer_momentum = None
        self.train_model_epoch = None
        self.train_model_batch_size = None
        self.save_model_name = None
        self.train_loss_log = None
        
        self.train_accuracy_log = None
        self.train_accuracy_5_log = None
        self.val_accuracy_log = None
        self.val_accuracy_5_log = None
        
        self.train_time = None
        self.train_time_epoch = None
        self.train_prediction_class = None
        self.train_prediction_5_class = None
        self.train_prediction_label = None
        self.val_prediction_class = None
        self.val_prediction_5_class = None
        self.val_prediction_label = None
        
    def remove_all_trained_models(self, setup): 
        all_saved_model = [f.name for f in os.scandir(f"{setup.path}/trained_models") if f.is_dir()  and f.name != '__pycache__' and f.name != '.ipynb_checkpoints']
        print (f"All saved models: {all_saved_model}")
        i = input("Press enter to confirm REMOVE ALL TRAINED MODELS or enter anything to stop. ")
        if (i == ""):
            for m in all_saved_model: 
                shutil.rmtree(f"{setup.path}/trained_models/{m}", ignore_errors=True)
            print ("All trained models have been REMOVED. ")
        else: 
            print ("User stopped removing all trained models. ")
    
class model:
    
    def __init__ (self, setup, dataset, model, loss_function, optimizer, learning_rate, momentum, epoches, batch_size, model_name, continuous): 
        
        self.dataset = dataset
        self.model_name = model
        self.model_loss_function_name = loss_function
        self.model_optimizer_name = optimizer
        self.model_optimizer_lr = learning_rate
        self.model_optimizer_momentum = momentum
        self.train_model_epoch = epoches
        self.train_model_batch_size = batch_size
        self.save_model_name = model_name
        self.continuous = continuous
        self.root = setup.path
        clear_output(wait=True)

        self.save_model_path = f"{setup.path}/trained_models/{self.save_model_name}"
        if (os.path.exists(self.save_model_path)): 
            if (not continuous): 
                all_saved_model = [f.name for f in os.scandir(f"{setup.path}/trained_models") if f.is_dir()  and f.name != '__pycache__' and f.name != '.ipynb_checkpoints']
                all_saved_model = sorted(all_saved_model)
                print (f"All saved models: {all_saved_model}")
                print (f"The saved model {self.save_model_name} already exists. Are you sure to overwrite it? ")
                i = input("Press enter to conform or enter anything to quit the training. ")
                if (i != ""): 
                    print ("User quit the model training. ")
                    return 1; 
            shutil.rmtree(self.save_model_path, ignore_errors=True)

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
        print ("All model training settings are completed. \n")
        print (f"Model name: {self.save_model_name}")
        print (f"Model: {self.model_name} (output classes = 17)")
        print (f"Loss function: {self.model_loss_function_name}")
        print (f"Optimizer: {self.model_optimizer_name} (learning rate={self.model_optimizer_lr}, momentum={self.model_optimizer_momentum})")
        print ("\nFor the model training: ")
        print (f"Epoches: {self.train_model_epoch}")
        print (f"Batch size: {self.train_model_batch_size}")
        if torch.cuda.is_available(): print (f"Device: GPU")
        else: print (f"Device: CPU")
        print ("\nThe training will start in 2 seconds ...")
        import time
        time.sleep (2)

        self.train_model (setup, dataset)
        if (not self.continuous): 
            self.print_train_result(setup)
        self.save_trained_model(setup)

    def print_train_val (self, option, epoch, batch, setup): 
        
        clear_output(wait=True)
        if (option == "train"): 
            print (f"Training the model {self.save_model_name} ({self.model_name}) (lr = {self.model_optimizer_lr}, batch_size = {self.train_model_batch_size})")
        if (option == "val"): 
            print (f"Validating the model {self.save_model_name} ({self.model_name}) (lr = {self.model_optimizer_lr}, batch_size = {self.train_model_batch_size})")
        if (option == "finish"): 
            print (f"Finished training {self.save_model_name} ({self.model_name}) (lr = {self.model_optimizer_lr}, batch_size = {self.train_model_batch_size})")
        print (f"Total Time: {math.floor(time.perf_counter() - self.start_time) // 60}m {math.floor(time.perf_counter() - self.start_time) % 60}s")
        
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
            print (f"Latest val accuracy: {self.val_accuracy_log[epoch - 1]}")
            print (f"Latest val_5 accuracy: {self.val_accuracy_5_log[epoch - 1]}")
        
    
    def train_model (self, setup, dataset): 
        
        self.train_loss_log = []
        
        self.train_accuracy_log = []
        self.train_accuracy_5_log = []
        self.val_accuracy_log = []
        self.val_accuracy_5_log = []
        
        self.train_prediction_class = []
        self.train_prediction_5_class = []
        self.train_label_class = []
        
        self.val_prediction_class = []
        self.val_prediction_5_class = []
        self.val_label_class = []
        
        clear_output(wait=True)
        print (f"Start training the model {self.save_model_name}. ")
        print (f"   -> After training, the trained model will appear in the trained_models folder. ")
        print (f"   -> There will be another .zip file of the trained model. You may DOWNLOAD and save that. ")
        print (f"   -> Upload .zip models downloaded to the trained_model folder, and unzip them later. ")
        
        import time
        time.sleep (3)
        print (f"\nLoading shuffled train and val dataset from {dataset.dataset}")
        self.train_dataloader = torch.utils.data.DataLoader (dataset = dataset.train_dataset, batch_size = self.train_model_batch_size, shuffle = True)
        self.val_dataloader = torch.utils.data.DataLoader (dataset = dataset.val_dataset, batch_size = self.train_model_batch_size, shuffle = True)

                          
        import time
        self.start_time = time.perf_counter()
        
        for epoch in range (self.train_model_epoch): 
            
           
            # Training
            self.model.train()
            
            correct_predictions = 0
            correct_predictions_5 = 0
            
            total_predictions = 0
            
            model_predictions = []
            model_predictions_5 = []
            
            model_labels = []
            
            training_loss_total = 0
            for batch, data in enumerate(self.train_dataloader, start = 0): 
                images, labels = data[0].to(self.device), data[1].to(self.device)
                self.model_optimizer.zero_grad()
                train_prediction = self.model(images)
                
                _, train_prediction_class = torch.max(train_prediction.data, dim=1)
                _, train_prediction_5_class = torch.topk(train_prediction.data, 5, dim=1)
                
                model_predictions.append(train_prediction_class)
                model_predictions_5.append(train_prediction_5_class)
                model_labels.append(labels)
                
                correct_predictions += (train_prediction_class == labels).sum().item()
                for i, x in enumerate(labels): 
                        if (labels[i].item() in train_prediction_5_class[i]): 
                            correct_predictions_5 += 1
                
                # print ("Prediction:", train_prediction_class)
                # print ("Labels:", labels)
                
                total_predictions += labels.size(0)
                
                training_loss = 0
                # if (self.model_name == "Inception_V1"): 
                #     training_loss = self.model_loss_function(train_prediction[0], labels)
                #     + 0.3 * self.model_loss_function(train_prediction[1], labels)
                #     + 0.3 * self.model_loss_function(train_prediction[2], labels)    
                # else: 
                training_loss = self.model_loss_function(train_prediction, labels)
                training_loss.backward()
                self.model_optimizer.step()
                training_loss_total += training_loss.item()
                
                self.print_train_val ("train", epoch, batch, setup)
                
            self.train_loss_log.append(training_loss_total)
            
            accuracy = correct_predictions / total_predictions
            accuracy_5 = correct_predictions_5 / total_predictions
            
            # print(model_predictions)
            model_predictions = torch.cat(model_predictions)
            # print(model_predictions_5)
            model_predictions_5 = torch.cat(model_predictions_5)
            model_labels = torch.cat(model_labels)
            
            self.train_accuracy_log.append(accuracy)
            self.train_accuracy_5_log.append(accuracy_5)
            
            self.train_prediction_class.append(model_predictions)
            self.train_prediction_5_class.append(model_predictions_5)
            self.train_label_class.append(model_labels)            
            
            
            # Validation
            self.model.eval()
            correct_predictions = 0
            correct_predictions_5 = 0
            
            total_predictions = 0
            
            model_predictions = []
            model_predictions_5 = []
            
            model_labels = []
            with torch.no_grad():
                for batch, data in enumerate(self.val_dataloader, start = 0): 
                    images, labels = data[0].to(self.device), data[1].to(self.device)
                    val_prediction = self.model(images)
                    
                    # output_softmax = [nn.functional.softmax(i, dim=0) for i in output]
                    # print(val_prediction)
                    _, val_prediction_class = torch.max(val_prediction.data, dim=1)
                    _, val_prediction_5_class = torch.topk(val_prediction.data, 5, dim=1)
                    
                    model_predictions.append(val_prediction_class)
                    model_predictions_5.append(val_prediction_5_class)
                    
                    model_labels.append(labels)
                    
                    correct_predictions += (val_prediction_class == labels).sum().item()
                    for i, x in enumerate(labels): 
                        if (labels[i].item() in val_prediction_5_class[i]): 
                            correct_predictions_5 += 1
                    
                    # print ("Prediction:", val_prediction_class)
                    # print ("Labels:", labels)
                    
                    total_predictions += labels.size(0)
                    self.print_train_val ("val", epoch, batch, setup)
                    
            accuracy = correct_predictions / total_predictions
            accuracy_5 = correct_predictions_5 / total_predictions
            
            model_predictions = torch.cat(model_predictions)
            model_predictions_5 = torch.cat(model_predictions_5)
            model_labels = torch.cat(model_labels)
            
            self.val_accuracy_log.append(accuracy)
            self.val_accuracy_5_log.append(accuracy_5)
            
            self.val_prediction_class.append(model_predictions)
            self.val_prediction_5_class.append(model_predictions_5)
            
            self.val_label_class.append(model_labels)
            
        
        self.print_train_val ("finish", self.train_model_epoch, None, setup)
        self.end_time = time.perf_counter()
        self.train_time = self.end_time - self.start_time
        

    def print_train_result(self, setup): 
        
        # x-axis: epoch
        # y-axis: train loss & val accuracy
        
        loss = []
        accuracy = []
        for i in self.train_loss_log: 
            loss.append(round(i, 2))
        for i in self.val_accuracy_log: 
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
        plot2_x = list(range(1, len(self.val_accuracy_log) + 1))
        plot2_y = self.val_accuracy_log
        plot2_legend, = plot2.plot(plot2_x, plot2_y, color = 'g', label = "Accuracy")

        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
        legends=[plot1_legend, plot2_legend]
        plot1.legend(handles=legends, loc=1)
        
        plt.savefig(f"{setup.path}/figures/[Train] model={self.model_name}  loss_function={self.model_loss_function_name}  optimizer={self.model_optimizer_name}  learning_rate={self.model_optimizer_lr}  momentum={self.model_optimizer_momentum}  epoches={self.train_model_epoch}  batch_size={self.train_model_batch_size}.pdf", format="pdf", bbox_inches="tight")
        
    def save_trained_model(self, setup):
        
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        
        os.makedirs(self.save_model_path, exist_ok = True)     
        f = open(f"{self.save_model_path}/model_description.txt", "w")
                          
        f.write(f"[Model Info]\n")
        f.write(f"Model_name: {self.save_model_name}\n")
        f.write(f"Model: {self.model_name}\n\n")
                          
        f.write(f"[Model Training Configurations]\n")
        f.write(f"Dataset: {self.dataset.dataset}\n")
        f.write(f"Loss function: {self.model_loss_function_name}\n")
        f.write(f"Optimizer: {self.model_optimizer_name}\n")
        f.write(f"Learning rate: {self.model_optimizer_lr}\n")
        f.write(f"Momentum: {self.model_optimizer_momentum}\n")
        f.write(f"Epoches: {self.train_model_epoch}\n")
        f.write(f"Batch size: {self.train_model_batch_size}\n\n")
        f.write(f"[Model Training Results]\n")    
        f.write(f"Train loss log: {self.train_loss_log}\n")
        
        f.write(f"Train accuracy log: {self.train_accuracy_log}\n")
        f.write(f"Train accuracy 5 log: {self.train_accuracy_5_log}\n")
        f.write(f"Val accuracy log: {self.val_accuracy_log}\n")
        f.write(f"Val accuracy 5 log: {self.val_accuracy_5_log}\n")
        
        f.write(f"Train time (s): {round(self.train_time, 2)}\n")
        self.train_time_epoch = round(self.train_time/self.train_model_epoch, 2)
        f.write(f"Time per epoch (s): {self.train_time_epoch}\n")
        f.write("")

        torch.save (self.train_prediction_class, f"{self.save_model_path}/train_predictions.pt")
        torch.save (self.train_prediction_5_class, f"{self.save_model_path}/train_predictions_5.pt")
        torch.save (self.train_label_class, f"{self.save_model_path}/train_labels.pt")
        torch.save (self.val_prediction_class, f"{self.save_model_path}/val_predictions.pt")
        torch.save (self.val_prediction_5_class, f"{self.save_model_path}/val_predictions_5.pt")
        torch.save (self.val_label_class, f"{self.save_model_path}/val_labels.pt")
        
        f.write(f"Model train predictions: {self.train_prediction_class}\n")
        f.write(f"Model train labels: {self.train_label_class}\n")
        f.write(f"Model val predictions: {self.val_prediction_class}\n")
        f.write(f"Model val labels: {self.val_label_class}\n")
        
        f.close()
                                
        # Trained model
        f = open(f"{self.save_model_path}/model_state_dict.pt", "w")
        torch.save(self.model.state_dict(), f"{self.save_model_path}/model_state_dict.pt")
        f.close()
        
        # Automatically zip the model
        os.system (f"zip -r {self.save_model_path}.zip {self.save_model_path}")
                          
    def load_model(model, setup, trained_model_name): 
        
        print (model)
        
        model.root = setup.path
        model.load_model_path = f"{setup.path}/trained_models/{trained_model_name}"
        if (os.path.exists(model.load_model_path)): 
            try: 
                file_read = open(f"{model.load_model_path}/model_description.txt", "r")
                for line in file_read.readlines():
                    if (len(line) > 12 and line[0:12] == "Model_name: "): model.save_model_name = line[12:len(line) - 1]
                    if (len(line) > 7 and line[0:7] == "Model: "): model.model_name = line[7:len(line) - 1]
                    if (len(line) > 9 and line[0:9] == "Dataset: "): model.dataset = line[9:len(line) - 1]
                    if (len(line) > 15 and line[0:15] == "Loss function: "): model.model_loss_function_name = line[15:len(line) - 1]
                    if (len(line) > 11 and line[0:11] == "Optimizer: "): model.model_optimizer_name = line[11:len(line) - 1]
                    if (len(line) > 15 and line[0:15] == "Learning rate: "): model.model_optimizer_lr = float(line[15:len(line) - 1])
                    if (len(line) > 10 and line[0:10] == "Momentum: "): 
                        if (line[10:len(line) - 1] != "None"): 
                            model.model_optimizer_momentum = float(line[10:len(line) - 1])
                        else: 
                            model.model_optimizer_momentum = None
                    if (len(line) > 9 and line[0:9] == "Epoches: "): model.train_model_epoch = int(line[9:len(line) - 1])
                    if (len(line) > 12 and line[0:12] == "Batch size: "): model.train_model_batch_size = int(line[12:len(line) - 1])
                    if (len(line) > 16 and line[0:16] == "Train loss log: "): model.train_loss_log = eval(line[16:len(line) - 1])
                    if (len(line) > 20 and line[0:20] == "Train accuracy log: "): model.train_accuracy_log= eval(line[20:len(line) - 1])
                    if (len(line) > 22 and line[0:22] == "Train accuracy 5 log: "): model.train_accuracy_5_log= eval(line[22:len(line) - 1])
                    if (len(line) > 18 and line[0:18] == "Val accuracy log: "): model.val_accuracy_log= eval(line[18:len(line) - 1])
                    if (len(line) > 20 and line[0:20] == "Val accuracy 5 log: "): model.val_accuracy_5_log= eval(line[20:len(line) - 1])
                    if (len(line) > 16 and line[0:16] == "Train time (s): "): model.train_time = float(eval(line[16:len(line) - 1]))
                    if (len(line) > 20 and line[0:20] == "Time per epoch (s): "): model.train_time_epoch = float(eval(line[20:len(line) - 1]))

                    # model.train_prediction_class = torch.load(f"{model.load_model_path}/train_predictions.pt")
                    # model.train_prediction_5_class = torch.load(f"{model.load_model_path}/train_predictions_5.pt")
                    # model.train_label_class = torch.load(f"{model.load_model_path}/train_labels.pt")
                    # model.val_prediction_class = torch.load(f"{model.load_model_path}/val_predictions.pt")
                    # model.val_prediction_5_class = torch.load(f"{model.load_model_path}/val_predictions_5.pt")
                    # model.val_label_class = torch.load(f"{model.load_model_path}/val_labels.pt")

                file_read.close()
                try: 
                    exec(f"import models.{model.model_name} as {model.model_name}")
                    
                    model.model = eval(f"{model.model_name}.{model.model_name} ()")
                    
                    # model.model.load_state_dict(torch.load(f"{model.load_model_path}/model_state_dict.pt"))
                    # model.model.eval()
                    # os.system("clear")

                    print (f"Load model {trained_model_name} ({model.model_name}) successfully.")
                    
                    return model; 
                    
                    
                except Exception: raise Exception (f"[Error] Load model from trained_models/{trained_model_name} fails! ") 
            except Exception: raise Exception (f"[Error] trained_models/{trained_model_name}/model_description.txt does not exist! ")         
        else: raise Exception (f"[Error] trained_models/{trained_model_name} does not exist! ")
        
        