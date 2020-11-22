#REFERENCES _ https://towardsdatascience.com/introduction-to-style-transfer-with-pytorch-339ba2219621
#importing torch and torch nn and numpy all for scietific computations
import torch
import torch.nn as nn
import numpy as np

#importing toch optim module for implementing the optimisation algorithms
import torch.optim as optim

#importing PIL and matplotlib to deal with Images(loading and displaying the images)
from PIL import Image
import matplotlib.pyplot as plt

#import torch transform to tranform pil image to torch tensor
import torchvision.transforms as transforms

#import torchvision.models to use pretrained models
import torchvision.models as models

#copy to deep copy the models
import copy


#If Nvedia GPU available use the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#if no gpu available size is small else big size of 512 * 512 pixels
imsize = 512 if torch.cuda.is_available() else 128



#Helper functions

#converting the normal images to tensors and loading it to GPU
def image_loader(image_name):
    # Used to transform image to the desired size and also convert them to pytorch tensors
    loader = transforms.Compose([transforms.Resize(imsize) , transforms.ToTensor()])
    #Using PIL to open the image 
    image = Image.open(image_name)
    #here unsqueeze is used so to add a fake dimensions to image at axis 0 as to match the networks input 
    image = loader(image).unsqueeze(axis=0)
    #adding image to gpu 
    return image.to(device , torch.float)


#converting the tensor into normal image to loading it to CpU for matplotlib
def imshow(tensor , title=None):
    #Create a unloader to convert tensor to actual PIL image 
    unloader = transforms.ToPILImage()
    #cloning to cpu memory and not giving the entire tensor 
    image = tensor.cpu().clone()
    #removing the fake dimension added
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(2)
    plt.close()

# get the features for the layers as layers listed in the paper
def get_features(model , image):

   #CONV4_2 WILL BE USED TO DESCRIBE THE CONTENT AND REST ALL FEATURES WE ARE CALCULATING WILL BE USED TO DESCRIBE THE STYLE
#THIS IS BECAUSE THE SHALLOWER LAYERS PRESERVE MORE DETAILS AND DEEPER LAYERS DISCARD THE DETAILS SO WE WANT TO EXRACT THE CONTENT FROM DEEPER LAYER
    layers = {
        '0' : 'conv1_1',
        '5' : 'conv2_1',
        '10' : 'conv3_1',
        '19' : 'conv4_1',
        '21' : 'conv4_2',
        '28' : 'conv5_1'
    }
    features = {}
    x = image
    #Passing the image through various layers(forward pass)
    for name , layer in model._modules.items():
        x = layer(x)
        #Appending the features of the layer if layer belongs to the dictionary 
        if name in layers:
            features[layers[name]] = x
    return features

#DEFINING THE GRAM MATRIX 
def gram(tensor):
    #REshaping the image channel tensor , with number of channels because index 0 in shape refers to batch size and 1 refers to number of channels
    t = tensor.view(tensor.shape[1] , -1)
    #ONCE THE TENSOR IS MADE INTO ROW MATRIX WE CAN RETURN THE T * T.transpose()
    return t @ t.T

#c_features and t_features are output of get_features when applied to content image and generated image respectively 
def content_loss(c_features, t_features):
    #This function returns the element wise difference of the channels and gives us the mean of all values then 
    loss = 0.5 * (c_features['conv4_2']  - t_features['conv4_2']) ** 2
    return torch.mean(loss)

def style_loss(s_grams , t_features , weights):
    #for each layer get style and target gramiams and compare 
    loss = 0
    for layer in weights:
        _ , d , h, w = t_features[layer].shape
        t_gram = gram(t_features[layer])

        layer_loss = (torch.mean((t_gram - s_grams[layer]) ** 2)) / d* h * w
        loss += layer_loss * weights[layer]
    return loss 




#Loading the PreTrained MOdel and Discarding the Deep layers of the network
model = models.vgg19(pretrained=True).features
model.to(device)


#Freezing the parameter so they are not trained of vgg19 model in backward prop 
for parameter in model.parameters():
    parameter.requires_grad_(False)







#IMPORTING THE STYLE AND CONTENT IMAGE
style_img  = image_loader("starry.jpg")
content_img = image_loader("harold.jpg")
# assert style_img.size() == content_img.size() , "Style and Content Images Must be of same size"

#PLOTTING THE STYLE AND CONTENT IMAGE
#interactive mode
plt.ion()
plt.figure()
imshow(content_img , title="ContentImage")
plt.figure()
imshow(style_img , title="StyleImage")

target = content_img.clone().requires_grad_(True).to(device)

style_weights = {

    'conv1_1': 0.2,
    'conv2_1': 0.2,
    'conv3_1' : 0.2 ,
    'conv4_1' : 0.2,
    'conv5_1' : 0.2
}


show = 1000
steps = 6000
alpha = 1
beta = 0.01


# Get the networks output for the style and content image 
s_features = get_features(model , style_img)
c_features = get_features(model , content_img)

#computing the gramiam for style image as a dictionary
s_grams = {layer : gram(features) for layer, features in s_features.items()}

#set optimiser to update target image pixels 
opt = optim.Adam([target] , lr = 0.1)

for step in range(0 , steps + 1):
    #set the gradients to 0
    opt.zero_grad()
    #get the value of t_features for different layer
    t_features = get_features(model , target)

    #get content and style loss
    c_loss = content_loss(c_features , t_features)
    s_loss = style_loss(s_grams , t_features , style_weights)

    #compute total loss and update the parameters
    total_loss = alpha * c_loss + beta * s_loss
    total_loss.backward()
    opt.step()

    #occassionally show the target image
    print("Painting the image : {} steps done , Please be patient".format(step) )
    if step % 50 == 0:
        print("Total loss:" , total_loss.item())
        imshow(target)
    if step % show == 0:
        #Create a unloader to convert tensor to actual PIL image 
        unloader = transforms.ToPILImage()
        #cloning to cpu memory and not giving the entire tensor 
        image_to_save = target.cpu().clone()
        #removing the fake dimension added
        image_to_save = image_to_save.squeeze(0)
        image_to_save = unloader(image_to_save)
        filename = "image_step_{}_1.jpg".format(step)
        print("Saving the image")
        image_to_save.save(filename)
