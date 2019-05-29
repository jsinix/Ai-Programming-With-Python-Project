# All the imports
import matplotlib.pyplot as plt
import torch, time, sys
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json, os, argparse 

# CLI inputs
def process_arguments(args):
    parser = argparse.ArgumentParser(description="Command Line App For Image Classifier")
    parser.add_argument('-a',
                        '--arch',
                        default='densenet121',
                        choices=['densenet121','vgg16'],
                        help="Architecture to use from torchvision.models [Available: densenet121, vgg16]"
                        )
    parser.add_argument('-e',
                        '--epochs',
                        required=True,
						type=int,
                        help="Number Of Epochs"
                        )
    parser.add_argument('-l',
                        '--learning_rate',
                        type=float,
						default=0.003,
                        help="Learning Rate For Model"
                        )
    parser.add_argument('-u',
                        '--hidden_units',
                        required=True,
                        type=int,
                        help="Number Of Hidden Units"
                        )	
    parser.add_argument('-s',
                        '--save_dir',
                        #required=True,
                        default='/trained_model_jasmeet/',
                        help="Directory To Save Checkpoints"
                        )	    
    parser.add_argument('-g',
                        '--gpu',
                        action='store_true',
                        help="Device To Use i.e. CPU or GPU"
                        )
    options = parser.parse_args(args)
    return vars(options)

if len(sys.argv) < 2:
    process_arguments(['-h'])
userOptions = process_arguments(sys.argv[1:])
print (userOptions)

data_dir = 'flowers'
cwd = os.getcwd()
checkpoint_dir = str(cwd) + userOptions.get('save_dir')

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Datasets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
#image_datasets = 
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
#dataloaders = 
trainloader = torch.utils.data.DataLoader(train_data, batch_size=96, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=96)
testloader = torch.utils.data.DataLoader(test_data, batch_size=96)

# cat to name mapping
with open('/home/workspace/aipnd-project/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if userOptions.get('gpu')==True else "cpu")
hidden_value = userOptions.get('hidden_units')

# prebuilt model classifiers to import from torchview
densenet_class = nn.Sequential(nn.Linear(1024, hidden_value),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 #nn.Linear(512, 256),
                                 #nn.ReLU(),
                                 #nn.Dropout(0.2),
                                 nn.Linear(hidden_value, 102),
                                 nn.LogSoftmax(dim=1))  

vgg_class = nn.Sequential(nn.Linear(25088, 12544),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(12544, 6272),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(6272, hidden_value),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(1024, 102),                            
                                 nn.LogSoftmax(dim=1))    
    
if userOptions.get('arch') == 'densenet121':
    model = models.densenet121(pretrained=True)
    model.classifier = densenet_class
elif userOptions.get('arch') == 'vgg16':
    model = models.vgg16(pretrained=True)
    model.classifier = vgg_class
    
for param in model.parameters():
    param.requires_grad = False
    
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=userOptions.get('learning_rate'))
model.to(device);

# Some variables 
epochs = userOptions.get('epochs')
steps = 0
running_loss = 0
print_every = 5
print("Using Device: {}".format(device))
train_losses, valid_losses = [], []

# Train the model while validating 
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
            
    else:
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        model.train()
        train_losses.append(running_loss/len(trainloader))
        valid_losses.append(test_loss/len(validloader))
        
        print("Epoch: {}/{}.. ".format(epoch+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))
        
print ("Complete") 

# testing the model 
# TODO: Do validation on the test set
test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
                    
        test_loss += batch_loss.item()
                    
        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
        print(f"Epoch {epoch+1}/{epochs}.. "
            f"Train loss: {running_loss/print_every:.3f}.. "
            f"Test loss: {test_loss/len(testloader):.3f}.. "
            f"Test accuracy: {accuracy/len(testloader):.3f}")

# TODO: Save the checkpoint 
checkpoint = {'input_size': 1024,
              'output_size': 102,
              'epochs': epochs,
              'batch_size': 96,
              'model': models.densenet121(pretrained=True),
              'classifier': model.classifier,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': train_data.class_to_idx
             }
torch.save(checkpoint, checkpoint_dir+'checkpoint_jasmeet.pth')
