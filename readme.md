# Glioma Growth Visualization Project
## Why?
Gliomas, a class of brain tumors, pose significant diagnostic and treatment challenges due to their diffusive growth patterns, making distinguishing between healthy and malignant tissue quite difficult. Accurate prediction of glioma progression is essential for optimizing treatment strategies and improving patient outcomes. 

## What?
This project aims to accurately model glioma growth using two types of well-established methods: a biological equation and machine learning. A GUI is provided to ensure the two methods are integrated and comparison of results is easy.

## How?
This project develops an integrated framework for glioma growth prediction by combining a reaction-diffusion mathematical model with a deep learning-based Convolutional Neural Network (CNN). The reaction-diffusion model simulates the biological mechanisms governing tumor cell proliferation and diffusion, while the CNN, leveraging a fine-tuned ResNet-50 architecture, classifies and predicts glioma growth patterns from MRI scans.

## Who?
We are a team of 5 working on developing this tool for our final year project.

## Methods
### Diffusion-Reaction Equation Based Prediction
The equation is primarily made up of three different biological factors:
1. Diffusion: The spatial spread of tumor cells through different brain tissue, modelled using the spatial gradient of the tumor cell density.
2. Proliferation: The local expansion of tumor cells over time due to mitosis, cell reproduction, modelled using a logistic growth function primarily governed by a reaction rate.
3. Spatial Decay: Limited growth potential as tumor cells spread, modelled through an exponential attenuation factor that depends on the distance from the tumor boundary and applies to both diffusion and reaction.

### Machine Learning Based Prediction 
The CNN is optimized for medical imaging through the following adaptations: 
1. Grayscale MRI Preprocessing: Since ResNet-50 expects three-channel input, the single grayscale channel is replicated across all three channels.
2. Transfer Learning: The pre-trained layers of ResNet-50 remain frozen to retain learned features, while additional fully connected layers fine-tune the model for glioma classification.
3. Dataset & Training: The model is trained on longitudinal MRI scans, employing data augmentation, normalization, and resizing to improve generalization. Training strategies include early stopping, learning rate scheduling, and model checkpointing to prevent overfitting.

## How to Run?
1. Install any Python IDE (VS Code, Pycharm etc) and Python 3
2. Download the repository
3. Open the repository inside Python IDE
4. Navigate to the project directory containing the file in a terminal and run pip install -r requirements.txt
5. Run app_main.py

## Conclusions
### Reaction-diffusion Model
The best-performing parameter combination across all tests was Dw=0.088, Dg=0.0176. 91% of tumors modelled achieved a DSC of 0.6 or higher. 46% of tumors reached a DSC of 0.8 or higher.
Overall, we observe that models using relatively low diffusion rates (i.e., all combinations with Dw ≤ 0.14) achieved higher mean DSC values (approximately 0.74–0.76) and exhibited narrower IQRs, indicating consistent performance across the central 50% of patients.
These findings suggest that with careful selection of biologically plausible diffusion parameters, the model can produce consistent and generally reliable tumor growth predictions.

### ML
The 3D U-Net architecture demonstrated significant improvements over the initial ResNet-50 approach, achieving a DSC of 0.82 and sensitivity of 0.84 on the validation dataset. 
When analyzed with the reaction-diffusion equations, a DSC mean of 0.3252 was achieved.

## Dataset
This project could not be possible without the [University of California San Francisco Adult Longitudinal Post-Treatment Diffuse Glioma (UCSF-ALPTDG) MRI Dataset](https://imagingdatasets.ucsf.edu/dataset/2).
112 total patients were used for modelling and testing purposes, all with increasing tumors.
The breakdown of tumor stages for 106 patients (not all were graded) is as follows:
- Grade 1: 1 patients 
- Grade 2: 14 patients 
- Grade 3: 21 patients 
- Grade 4: 70 patients 

