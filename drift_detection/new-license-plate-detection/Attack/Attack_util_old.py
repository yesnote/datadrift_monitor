import torch
import cv2
import datetime
from pathlib import Path
from OD_models.CNN_util import CNN_resnet_50
from OD_models.Transformer_util import Vision_transformer
from Data.Data_util import data_util
import torchvision.transforms as transforms

from torchattacks import *

"""
Module for implements attacks using torchattack lib, 

MIT License

Copyright (c) 2020 Harry Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Implementation in colab link: https://colab.research.google.com/drive/1ufjcqUiMf0eajV0Fq1IZKMqZazn52rQt?usp=sharing
"""


class attack_util():
    def __init__(self, dataset_name,model_name):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = self.upload_dataset(dataset_name)
        self.target_model = self.upload_model(model_name)
        self.conf = self.attack_config()
        self.output_path = self.create_output_path(dataset_name,model_name)

    def upload_dataset(self, dataset_name):
        """
        Function for loading the dataset using the data_util module.
        :return: list of PIL images.
        """
        images, labels = data_util.load_image_label_dataset(dataset_name)
        return tuple(zip(images, labels))

    def upload_model(self, model_name):
        if model_name=="Vision Transformer":
            vit = Vision_transformer()
            return vit.export_model()
        # default is CNN resnet 50
        else:
            cnn_resnet_50 = CNN_resnet_50()
            return cnn_resnet_50.export_model()

    def create_output_path(self,dataset_name,model_name):
        time = datetime.datetime.now().strftime("%d-%m-%Y_%H;%M")
        output_path = f"Output/{model_name}-{dataset_name}-{time}"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        return output_path

    def attack_config(self):
        """
        :param model: req. pytorch pre trained model.
        :param device: req. pytorch device (cpu or gpu)
        :return: attacks name and params.
        """
        attacks = [
            FGSM(self.target_model.to(self.device), eps=100 / 255),
            BIM(self.target_model, eps=80 / 255, alpha=2 / 255, steps=1000),
            PGD(self.target_model, eps=120 / 255, alpha=2 / 225, steps=1000, random_start=True),
            CW(self.target_model, c=2, lr=0.01, steps=3000, kappa=0),
            RFGSM(self.target_model, eps=80 / 255, alpha=2 / 255, steps=1000),
        ]

        attacks_name = [
            'FGSM',
            'BIM',
            'PGD',
            'CW',
            'RFGSM',
        ]
        return dict(zip(attacks_name, attacks))

    def generate_attack(self,attack_name,save_samples = False,sample_label = None):
        attack_vector = self.conf[attack_name]
        if sample_label:
            adversarial_samples = self.attack_single_sample(self.device,attack_vector,sample_label)
        else:
            adversarial_samples = self.attack_dataset(self.device,attack_vector)
        if save_samples:
            self.save_samples(adversarial_samples)
    def transform(self,sample):
        transform_function=transforms.Compose([transforms.PILToTensor()])
        transformed_sample = transform_function(sample)/255
        transformed_sample = torch.unsqueeze(transformed_sample, 0)
        return transformed_sample

    def attack_single_sample(self, device, attack_vector,sample_label):
        adversarial_samples = []
        sample, label = sample_label
        sample_tensor = self.transform(sample)
        label_tensor = torch.tensor(label)
        sample, label = sample_tensor.to(device), label_tensor.to(device)
        adv_x = attack_vector(sample, label)
        adversarial_samples.append(adv_x)
        return adversarial_samples

    def attack_dataset(self, device, attack_vector):
        adversarial_samples = []
        for sample_label in self.dataset:
            adv_x = self.attack_single_sample(device,attack_vector,sample_label)
            adversarial_samples.append(adv_x)
        return adversarial_samples

    def save_samples(self, adversarial_samples):
        for idx,adv_x in enumerate(adversarial_samples):
            cv2.imwrite(f'{self.output_path}/adversarial_{idx}.jpg',adv_x)


    def demo(self):
        attack = 'FGSM'
        image_label = self.dataset[1]
        self.generate_attack(attack,save_samples = True,sample_label = image_label)

