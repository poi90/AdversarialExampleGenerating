import pandas as pd
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from zipfile import ZipFile
import os
import torchvision
import torchvision.transforms as T
from torch import nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torchvision.utils import save_image
import argparse
parser = argparse.ArgumentParser(description='Generate Adversarial Example')
parser.add_argument('--model_path', type=str, default='pytorch/vision:v0.6.0',
                    help='path to the model on web of pytorch')
parser.add_argument('--model_name', default='densenet121',type = str,
                    help='name of the model')
parser.add_argument('--steps', default=40, type=int,
                    help='perturb number of steps, (default: 20)')
parser.add_argument('--alpha', default=0.03, type=float)
parser.add_argument('--img_path', default="images/", type=str, help='path of the images')
parser.add_argument('--d_path', default="adImages/", type=str, help='path of goal folder')
parser.add_argument('--classes_path', default="classes.txt", type=str, help='path of the classes list')
parser.add_argument('--eps', default=2 * 8 / 225, type= float)
args = parser.parse_args()




trans = T.Compose([T.ToTensor(), T.Lambda(lambda t: t.unsqueeze(0))])

loss = nn.CrossEntropyLoss()


def load_image(img_path):
    img = trans(Image.open(img_path).convert('RGB'))
    return img


def get_class(img, inputModel, class_path):
    classes = eval(open(class_path).read())
    x = Variable(img, volatile=True)
    cls = inputModel(x).data.max(1)[1].cpu().numpy()[0]

    return classes[cls]


loss_object = tf.keras.losses.CategoricalCrossentropy()


def no_target_attack(img, per_steps, input_model, step_alpha, eps):
    label = torch.zeros(1, 1)
    x, y = Variable(img, requires_grad=True), Variable(label)

    for step in range(per_steps):
        zero_gradients(x)
        out = input_model(x)
        y.data = out.data.min(1)[1]

        _loss = loss(out, y)

        _loss.backward()
        normed_grad = step_alpha * torch.sign(x.grad.data)

        step_adv = x.data - normed_grad

        adv = step_adv - img

        adv = torch.clamp(adv, -eps, eps)
        result = img + adv
        result = torch.clamp(result, 0.0, 1.0)
        x.data = result
    return result, adv


def generate(resource_folder, destination_folder, input_model, step_alpha, steps, eps):
    for fileName in os.listdir(resource_folder):
        r_path = resource_folder + fileName

        r_image = load_image(r_path)
        d_path = destination_folder + fileName
        adv_img, noise = no_target_attack(r_image, steps, input_model, step_alpha, eps)
        save_image(adv_img, d_path)


if __name__ == "__main__":
    model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)

    #model = torch.hub.load(args.model_path, args.model_name, pretrained=True)
    model.eval()
    generate(args.img_path, args.d_path, model, args.alpha, args.steps, args.eps)

