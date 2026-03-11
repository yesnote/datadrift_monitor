# DiL: an Explainable and Actionable Metric for Abnormal Uncertainty in Object Detection

Code project for distinctive localization (DiL), an unsupervised explainable metric that captures the 
internal uncertainty of an OD model when faced with abnormal scenes.

![Alt text](samples_for_readme/Design.png?raw=true "Title")
 
## Description

We present DiL, an actionable explainable metric that can:
 
1) Reflect the model’s inner uncertainty, both quantitatively and visually.
2) Explain the DiL score produced. 
3) Be leveraged for preventive actions.

This project presents the code implementation of DiL's main ability -  to quantitatively interprets 
an abnormality’s effect on the object detection model’s decision-making process.

For more information, please see the paper draft.

## Getting started

In order to use Dil's full capabilities, please perform the following steps:
1) Create a new environment and install all required packages using the dil_requirements.txt file.
2) Download the object detection models from the following link and store the models in the models_weights project folder:
https://drive.google.com/file/d/17m4Q_XvrYVzRb0_MrG_97As1FmUYkepB/view?usp=sharing
3) Download the use cases datasets (clean, partial-occlusion, out-of-distribution and adversarial use cases 
of COCO and SuperStore datasets) from the following link and store them in the datasets project folder: 
https://drive.google.com/file/d/1EyYU_M33TZ3QAN40wqgLMMPrN-VImjau/view?usp=drive_link

4) The project files-tree should look like this after downloading the files and storing them in the correct location: 
![Alt text](samples_for_readme/files-tree.png?raw=true "Title")

5) Start a demo experiment using the Main.py module:
   1) Chose which evaluation space to apply DiL on in the experiment (digital-COCO, physical-SuperStore)
   2) Chose which use case to apply DiL on in the experiment (clean, partial occlusion, out-of-distribution, adversarial). 

6) Please explore the configuration file to change the following settings:
   1) Saliency map technique (GradCAM/GradCAM++/EigenCAM/EigenGradCAM/GradCAMElementWise, default is GradCAM)
   2) Models (Faster R-CNN/Grid R-CNN/Double heads R-CNN/Cascade R-CNN/Cascade RPN/YOLOv3/YOLOv5/YOLOF 
   default is Faster R-CNN
   3) The models decision thresholds.
