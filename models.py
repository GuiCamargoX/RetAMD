import os
absolute_path = os.path.dirname(__file__)

# The class containing the model
import torch
from PIL import Image
import torchvision
from torchvision import transforms
import torch.nn as nn
from fundus_prep import PreprocessEyeImages
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

import torch.backends.cudnn as cudnn
import argparse
import os
from densenet_mcf import dense121_mcs
from PIL import Image, ImageCms

class ResNet:
    def __init__(self):
        self.classes = ['Age-related macular degeneration (AMD) DETECTED', 'NEGATIVE for AMD']

        self.model = torchvision.models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

        self.model.load_state_dict(torch.load(os.path.join(absolute_path,'weights/resnet18_weight.pt'), map_location=torch.device('cpu') ))
        self.model.eval()

    
    def infer(self, image_path):
        input_image = Image.open(image_path)
        preprocess = transforms.Compose([
                PreprocessEyeImages(),
                transforms.Resize(390),
                transforms.CenterCrop(256),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        input_tensor = preprocess(input_image)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0) 

        # move the input and model to GPU for speed if available
        #if torch.cuda.is_available():
        #    input_batch = input_batch.to('cuda')
        #    self.model.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch)

        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        output = output.softmax(-1)
        confidence, index = torch.max(output[0], 0)

        #save prep
        prep_img, heat_img = self.gradCam( input_batch )

        return (self.classes[index.item()], confidence.item(), prep_img, heat_img)


    def gradCam(self, input_batch):
        #save prep
        img = input_batch.squeeze(0)
        inp = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        rgb_img = np.clip(inp, 0, 1)

        target_layer = self.model.layer4[-1]

        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=self.model, target_layer=target_layer)
        cam.batch_size = 82

        target_category = None

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_batch, target_category=target_category)

        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam)

        return Image.fromarray( (rgb_img * 255).astype(np.uint8) ), Image.fromarray( cam_image[:, :, ::-1] )


class EyeQ:

    def __init__(self):
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.classes = ['Good', 'Usable', 'Bad']

        # options
        cudnn.benchmark = True

        self.model = dense121_mcs(n_class= 3)
        loaded_model = torch.load(os.path.join(os.path.join(absolute_path,'weights/densenet_weight.tar')) , map_location=torch.device('cpu') )
        self.model.load_state_dict(loaded_model['state_dict'])

        self.model.to(self.device)

       

    def infer(self, image_path):
        image = Image.open(image_path)
        imagesA, imagesB, imagesC = self.preprocess(image)

        # Testing
        self.model.eval()
        
        imagesA = imagesA.to(self.device)
        imagesB = imagesB.to(self.device)
        imagesC = imagesC.to(self.device)

        # create a mini-batch as expected by the model
        imagesA = imagesA.unsqueeze(0)
        imagesB = imagesB.unsqueeze(0)
        imagesC = imagesC.unsqueeze(0)

        _, _, _, _, result_mcs = self.model(imagesA, imagesB, imagesC)
        

        pred= torch.argmax(result_mcs,1).cpu()
        return self.classes[ pred.item() ]
        

    
    def preprocess(self, sample):
        image = sample.convert('RGB')

        transform1 = transforms.Compose([
                PreprocessEyeImages(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
            ])
        image = transform1(image)


        srgb_profile = ImageCms.createProfile("sRGB")
        lab_profile = ImageCms.createProfile("LAB")
        rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
        
        img_hsv = image.convert("HSV")
        img_lab = ImageCms.applyTransform(image, rgb2lab_transform)

        img_rgb = np.asarray(image).astype('float32')
        img_hsv = np.asarray(img_hsv).astype('float32')
        img_lab = np.asarray(img_lab).astype('float32')


        transform2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ])
        img_rgb = transform2(img_rgb)
        img_hsv = transform2(img_hsv)
        img_lab = transform2(img_lab)

        
        return torch.FloatTensor(img_rgb), torch.FloatTensor(img_hsv), torch.FloatTensor(img_lab)

