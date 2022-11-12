
import torchvision
from torchvision import transforms

path_1 = path + "/Dataset"
path_training = path_1 + "/training"
path_validation = path_1 + "/validation"
path_testing = path_1 + "/testing"

Dataset_training = torchvision.datasets.ImageFolder(
    root = path_training, transform = 
    transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize([256,256]),
        transforms.RandomCrop([224, 224]), 
        transforms.RandomHorizontalFlip(0.5), 
        transforms.ToTensor(), 
        transforms.Normalize([0, 0, 0],[1.0, 1.0, 1.0])
    ]))
Dataset_validation = torchvision.datasets.ImageFolder(
    root = path_validation, transform = 
    transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize([256,256]),
        transforms.RandomCrop([224, 224]), 
        transforms.ToTensor(), 
        transforms.Normalize([0, 0, 0],[1.0, 1.0, 1.0])
    ]))
Dataset_testing = torchvision.datasets.ImageFolder(
    root = path_testing, transform = 
    transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize([256,256]),
        transforms.RandomCrop([224, 224]), 
        transforms.ToTensor(), 
        transforms.Normalize([0, 0, 0],[1.0, 1.0, 1.0])
    ]))

print ("There is a total of " + str(len(Dataset_training)) + " training images. ")
print ("There is a total of " + str(len(Dataset_validation)) + " validation images. ")
print ("There is a total of " + str(len(Dataset_testing)) + " testing images. ")


