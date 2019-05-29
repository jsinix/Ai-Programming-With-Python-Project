# References:
# https://knowledge.udacity.com/questions/32973
# https://knowledge.udacity.com/questions/35290
# 

import torch, time, sys
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json, os, argparse
from PIL import Image

def process_arguments(args):
    parser = argparse.ArgumentParser(description="Command Line App For Image Classifier")
    parser.add_argument('-k',
                        '--top_k',
                        default=1,
						type=int,
                        help="Top n values"
                        )
    parser.add_argument('-j',
                        '--json',
                        required=True,
                        help="JSON file that maps the class values"
                        )
    parser.add_argument('-i',
                        '--image',
                        required=True,
                        help="Input image to test the model"
                        )
    parser.add_argument('-c',
                        '--checkpoint',
                        required=True,
                        help="Saved checkpoint model"
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

test_img_path = userOptions.get('image')   
top_n_predictions = userOptions.get('top_k')
checkpoint_img = userOptions.get('checkpoint')
json_file = userOptions.get('json')
device = torch.device("cuda" if userOptions.get('gpu')==True else "cpu")

with open(json_file, 'r') as f:
    cat_to_name = json.load(f)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=device)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    for param in model.parameters():
        param.requires_grad = False
    return model, checkpoint['class_to_idx']

model, class_to_idx = load_checkpoint(checkpoint_img)

def process_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    pil_image = Image.open(image)
    np_image = preprocess(pil_image).numpy()
    return np_image

def predict(image_path, model, topk=top_n_predictions):
    image = process_image(image_path)
    image_t = torch.from_numpy(image).type(torch.FloatTensor).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logps = model.forward(image_t)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk)
        top_p = top_p.detach().numpy().tolist()[0]
        top_class = top_class.detach().numpy().tolist()[0]
    return top_p, top_class

prob_arr, pred_indexes = predict(test_img_path, model)
idx_to_class = {v: k for k, v in model.class_to_idx.items()}
pred_labels = [idx_to_class[x] for x in pred_indexes]
pred_class = [cat_to_name[str(x)] for x in pred_labels]
#returns numpy
image = process_image(test_img_path)
max_index = pred_indexes[0]

for cl, pro in zip(pred_class, prob_arr):
    print (f"{cl} : {pro}")
