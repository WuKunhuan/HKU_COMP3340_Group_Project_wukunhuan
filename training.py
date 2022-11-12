from IPython.display import clear_output
import math
def print_training_info (name, epoch, epoches, batch_size, batch, batches, training_losses, accuracies, accuracies_stat): 
    clear_output(wait=True)
    print (f"Training the model {name} (batch size = {batch_size})")
    print ("[Epoch] ", end = "")
    for i in range (0,  math.floor(epoch * 1.0 / epoches * 50), 1): 
        print ("█", end = "")
    for i in range (0,  50 - math.floor(epoch * 1.0 / epoches * 50), 1): 
        print ("░", end = "")
    print (f" {epoch} out of {epoches}")
    print ("[Batch] ", end = "")
    for i in range (0,  math.floor(batch * 1.0 / batches * 50), 1): 
        print ("█", end = "")
    for i in range (0,  50 - math.floor(batch * 1.0 / batches * 50), 1): 
        print ("░", end = "")
    print (f" {batch} out of {batches}")
    l = []
    for i in training_losses: 
        l.append(round(i, 2))
    print (f"[Training Loss] {l}")
    l = [(f"{accuracies[i]:.3f} ({accuracies_stat[i][0]}/{accuracies_stat[i][1]})") for i in range(len(accuracies))]
    print (f"[Accuracy] {l}")

def trained_model (name, epoches, batch_size, learning_rate, optimizer):
    model = None
    model_optimizer = None
    clear_output(wait=True)
    print (f"Loading the model {name}")
    if ('model_name_setting'):
        if (name == 'AlexNet'): model = AlexNet(num_classes = 17)
        elif (name == 'VGG_11'): model = VGG_11(num_classes = 17)
        elif (name == 'VGG_11_BN'): model = VGG_11_BN(num_classes = 17)
        elif (name == 'VGG_13'): model = VGG_13(num_classes = 17)
        elif (name == 'VGG_13_BN'): model = VGG_13_BN(num_classes = 17)
        elif (name == 'VGG_16'): model = VGG_16(num_classes = 17)
        elif (name == 'VGG_16_BN'): model = VGG_16_BN(num_classes = 17)
        elif (name == 'VGG_19'): model = VGG_19(num_classes = 17)
        elif (name == 'VGG_19_BN'): model = VGG_19_BN(num_classes = 17)
        elif (name == 'ResNet_18'): model = ResNet(17, BasicBlock, [2, 2, 2, 2])
        elif (name == 'ResNet_34'): model = ResNet(17, BasicBlock, [3, 4, 6, 3])
        elif (name == 'ResNet_50'): model = ResNet(17, Bottleneck, [3, 4, 6, 3])
        elif (name == 'Inception_V1'): model = Inception_V1(num_classes = 17)
        elif (name == 'Inception_V4'): model = Inception_V4(num_classes = 17)
        else: raise Exception(f"[Error] Invalid model name: {name}")
        model.to(device)
    if ('model_loss_function_setting'):
        model_loss_function = nn.CrossEntropyLoss()
    if ('model_optimizer_setting'):
        if (optimizer in ['Adam', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam']): 
            model_optimizer = {
                'Adam':torch.optim.Adam(model.parameters(), lr=learning_rate), 
                'SGD':torch.optim.SGD(model.parameters(), lr=learning_rate),
                'RMSprop':torch.optim.RMSprop(model.parameters(), lr=learning_rate), 
                'Adadelta':torch.optim.Adadelta(model.parameters(), lr=learning_rate),
                'Adagrad':torch.optim.Adagrad(model.parameters(), lr=learning_rate), 
                'Adamax':torch.optim.Adamax(model.parameters(), lr=learning_rate),
                'Nadam':torch.optim.NAdam(model.parameters(), lr=learning_rate), 
            }[optimizer]
        else: raise Exception(f"[Error] Invalid model optimizer: {optimizer}")
    model_training_losses = []
    model_accuracies = []
    model_accuracies_stat = []
    model_prediction_label = []
    print (f"Loading the training dataset ({len(Dataset_training)} images)")
    DataLoader_training = torch.utils.data.DataLoader (dataset = Dataset_training, batch_size = batch_size, shuffle = True)
    print (f"Loading the validation dataset ({len(Dataset_validation)} images)")
    DataLoader_validation = torch.utils.data.DataLoader (dataset = Dataset_validation, batch_size = batch_size, shuffle = True)
    print (f"Loading the testing dataset ({len(Dataset_testing)} images)")
    DataLoader_testing = torch.utils.data.DataLoader (dataset = Dataset_testing, batch_size = batch_size, shuffle = True)
    
    print_training_info (name, 0, epoches, batch_size, 0, len(DataLoader_training), model_training_losses, model_accuracies, model_accuracies_stat)
    for epoch in range(epoches): 
        # Training
        clear_output(wait=True)
        model.train()
        model_training_loss = 0
        for batch, data in enumerate(DataLoader_training, start = 0): 
            images, labels = data
            model_optimizer.zero_grad()
            model_training_output = model(images.to(device))
            training_loss = 0
            if (name == "Inception_V1"): 
                training_loss = model_loss_function(model_training_output[0], labels.to(device)) + 0.3 * model_loss_function(model_training_output[1], labels.to(device)) + 0.3 * model_loss_function(model_training_output[2], labels.to(device))    
            else: 
                training_loss = model_loss_function(model_training_output, labels.to(device))
            training_loss.backward()
            model_training_loss += training_loss.item()
            model_optimizer.step()
            print_training_info (name, epoch, epoches, batch_size, batch + 1, len(DataLoader_training), model_training_losses, model_accuracies, model_accuracies_stat)
        # Validation
        print (f"Validating the trained model after epoch {epoch + 1}")
        model.eval()
        model_validation_accuracy = 0
        model_correct_predictions = 0
        with torch.no_grad():
            for batch, data in enumerate(DataLoader_training, start = 0): 
                images, labels = data[0].to(device), data[1].to(device)
                model_output = model(images)
                model_output_softmax = [nn.functional.softmax(i, dim=0) for i in model_output]
                model_prediction = torch.max(model_output.data, dim=1)[1]
                model_correct_predictions = (model_prediction == labels).sum().item()
                l = []
                for i in range(len(model_prediction)): 
                    l.append((model_prediction[i], labels[i]))
            model_accuracy = model_correct_predictions / len(Dataset_validation)  
        model_training_losses.append(model_training_loss)
        model_accuracies.append(model_accuracy)
        model_accuracies_stat.append((model_correct_predictions, len(Dataset_validation)))
        model_prediction_label.append(l)
    print_training_info (name, epoches, epoches, batch_size, len(DataLoader_training), len(DataLoader_training), model_training_losses, model_accuracies, model_accuracies_stat)
    return (name, model_training_losses, model_accuracies, model_accuracies_stat, model_prediction_label)