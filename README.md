# Skin Lesion Classification: Cancer or Not?
**By: Sameeha Ramadhan**

## Summary 

The goal of this project is to build a tool to correctly identify if whether or not a skin lesion is cancerous. The model will take in a picture of said lesion and calculate the probability that said lesion is benign (non-cancerous) or malignant (cancer). 

The Neural Network chosen was the Convolutional Neural Network (CNN) as it is one of the preferred for image processing. In this project I use a standard CNN model without data augmentation, two 5-convolutional block models, an EfficientNet-B0 and a ResNet50 model.

### Business Problem:

The use of such an algorithm would assist doctors and medical professionals in decreasing the time it takes to review and diagnose a patient as well as benefitting the patient by knowing sooner than later if cancer is present so that treatment could begin promptly. In addition, if a person is in doubt about a particular mole but hesitant to visit the doctor, this tool could give peace of mind (by diagnosing as benign) or encourage an appointment (should it output malignant).

### What About Skin Cancer?

**(insert image)**

The skin is the body’s largest organ. Skin has several layers, but the two main layers are the epidermis (upper or outer layer) and the dermis (lower or inner layer). Skin cancer begins in the epidermis, which is made up of three kinds of cells—

**Squamous cells: Thin, flat cells that form the top layer of the epidermis.

**Basal cells: Round cells under the squamous cells.


**Melanocytes: Cells that make melanin and are found in the lower part of the epidermis. Melanin is the pigment that gives skin its color. When skin is exposed to the sun, melanocytes make more pigment and cause the skin to darken.

Basal and squamous cell carcinomas are the two most common types of skin cancer. They begin in the basal and squamous layers of the skin, respectively. Both can usually be cured, but they can be disfiguring and expensive to treat.

Melanoma, the third most common type of skin cancer, begins in the melanocytes. Of all types of skin cancer, melanoma causes the most deaths because of its tendency to spread to other parts of the body, including vital organs.

Most cases of skin cancer are caused by overexposure to ultraviolet (UV) rays from the sun, tanning beds, or sunlamps. UV rays can damage skin cells. In the short term, this damage can cause a sunburn. Over time, UV damage adds up, leading to changes in skin texture, premature skin aging, and sometimes skin cancer. UV rays also have been linked to eye conditions such as cataracts. [Source](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

### What is CNN?

A Convolutional Neural Network (CNN/ConvNet) is a Deep Learning algorithm which can take in an image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a CNN is much lower as compared to other classification algorithms. In simpler terms, the role of the ConvNet/CNN is to reduce the images into a form that is easier to process, without losing features that are critical for getting a good prediction. For an in-depth guide on CNNs, click [here](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

## The Data

### Obtaining the data:

The data was gathered from a number of sources including [dermascopy.org](https://sites.google.com/site/robustmelanomascreening/dataset), cancer.com, skincancer.org, cdc.gov, kaggle.com and more. I've compiled the dataset using a number of techniques including web scraping, and will upload the full dataset (contains approximately 28,000 images) to be available for download.

### Exploring the Data:
I loaded the data and explored the number of images per set and displayed a few to make sure that all of the files are readable.

(insert skin plot)

### Data Visualization:

Next I plotted the data to check the balance and verify that it is evenly distributed:

(insert df balance chart)


