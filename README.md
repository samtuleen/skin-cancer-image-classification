![R](https://user-images.githubusercontent.com/71333855/130537828-502583bb-e539-4a44-b04c-46887ac8addd.png)

# Skin Lesion Classification: Cancer or Not?
**By: Sameeha Ramadhan**

## Summary 

The goal of this project is to build a tool to correctly identify if whether or not a skin lesion is cancerous. The model will take in a picture of said lesion and calculate the probability that said lesion is benign (non-cancerous) or malignant (cancer). 

The Neural Network chosen was the Convolutional Neural Network (CNN) as it is one of the preferred for image processing. In this project I use a standard CNN model without data augmentation, two 5-convolutional block models, an EfficientNet-B0 and a ResNet50 model.

I've chosen this specific project because I've recently lost my grandfather on Memorial Day 2021 due to heart failure. He was a Purple Heart veteran of the Vietnam war, who unfortunately suffered the effects of Agent Orange (a defoliant and herbicide/chemical sprayed by the US in the Vietnam war). At the change of every season he would suffer severely from the long lasting effects that would cause his skin to peel off, as well as causing various lesions to form. He would be in so much agony that he'd ban any of his family from seeing him until the episode would pass. This went on till his untimely passing 3 months ago at the age of 76. 

One of the major health issues as a result of Agent Orange includes heart disease (which he ultimately succumbed to) as well as passing on the effects to one's offspring. Unfortunately, the skin flare ups were passed on to some of his decsendants, including my aunt and younger sister. I would hope to expand on this project in the near future to develop an algorithm that can identify the effects of Agent Orange and get the suffering the care that they need.

### Business Problem:

The use of such an algorithm would assist doctors and medical professionals in decreasing the time it takes to review and diagnose a patient as well as benefitting the patient by knowing sooner than later if cancer is present so that treatment could begin promptly. In addition, if a person is in doubt about a particular mole but hesitant to visit the doctor, this tool could give peace of mind (by diagnosing as benign) or encourage an appointment (should it output malignant).

### What About Skin Cancer?

![whatisskincancer](https://user-images.githubusercontent.com/71333855/130537852-73706040-663e-452a-b491-815a652727b0.jpg)

The skin is the body’s largest organ. Skin has several layers, but the two main layers are the epidermis (upper or outer layer) and the dermis (lower or inner layer). Skin cancer begins in the epidermis, which is made up of three kinds of cells—

**Squamous cells: Thin, flat cells that form the top layer of the epidermis.**

**Basal cells: Round cells under the squamous cells.**


**Melanocytes: Cells that make melanin and are found in the lower part of the epidermis. Melanin is the pigment that gives skin its color. When skin is exposed to the sun, melanocytes make more pigment and cause the skin to darken.**

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

![skincancerplot](https://user-images.githubusercontent.com/71333855/130537874-fa85b5d2-8a85-45dc-80fa-ccca7d8dd861.png)

### Data Visualization:

Next I plotted the data to check the balance and verify that it is evenly distributed:

![traintestbalance](https://user-images.githubusercontent.com/71333855/130538239-b4261d98-792d-4341-b668-1ce774d5234b.jpg)


Image: Training set (left) and test set (right).

As seen above, the data is fairly balanced. Given that large datasets are necessary for Deep Learning and this notebook is ran on a sample of the data, the data will be augmented so that the number of images increases to further stabilize the model.

## Image Preprocessing

I've used three processes for preparing the images for modeling:

### Image Rescaling:
All images need to be rescaled to a fixed size before feeding them to the neural network. The larger the fixed size, the less shrinking required, which means less deformation of patterns inside the image and in turn higher chances that the model will perform well. I've rescaled all of the images to 256 colors (0 - 255).

### Data Augmentation:
The performance of Deep Learning Neural Networks often improves with the amount of data available, therefore increasing the number of data samples could result in a more skillful model. Data Augmentation is a technique to artificially create new training data from existing training data. This is done by applying domain-specific techniques to examples from the training data that create new and different training examples by simply shifting, flipping, rotating, modifying brightness, and zooming the training examples. [Source](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)

I've augmented the data for my various models using a number of parameters, including:

** zoom_range=0.3

** vertical_flip=True

** width_shift_range=0.2 

** height_shift_range=0.2,

** horizontal_flip=True

** vertical_flip=True

# Modeling

After building and testing over 20 models I've settled on the following 5 as the ones with the best results: two 5-convolutional block models, one CNN model with no data augmentation, one EfficientNet-B0 model and one ResNet50 model. For my 5 convolutional block model I've done the following:

1- Built five convolutional blocks comprised of convolutional layers, BatchNormalization, and MaxPooling.

2- Reduced over-fitting by using dropouts.

3- Used ReLu (rectified linear unit) activation function for all except the last layer. Since this is a binary classification problem, Sigmoid was used for the final layer.

4- Used the Adam optimizer and binary-entropy for the loss.

5- Added a flattened layer followed by fully connected layers. This is the last stage in CNN where the spatial dimensions of the input are collapsed into the channel dimension.

Note:

** ReLu: a piecewise linear function that outputs zero if its input is negative, and directly outputs the input otherwise.
** Sigmoid: its gradient is defined everywhere and its output is conveniently between 0 and 1 for all x.


After testing all of my models, I've chosen the following since it has offered satisfactory results based on both its validation and test accuracy. The best performing model is a ResNet50 model, a pretrained image classification model that consists of a 50 layer deep convolutional network. I've chosen this model because since my dataset is large and this particular model has been trained on thousands of images in very deep layers. 

I've augmented the data using the following parameters: rescale = 1./255, rotation_range=45, zoom_range = 0.2, width_shift_range=0.2, and height_shift_range=0.2. Then I used softmax as my activation function to calculate the relative probabilities [Source](https://www.analyticsvidhya.com/blog/2021/04/introduction-to-softmax-for-neural-network/)

# Analyzing the Results:

The simplest way to analyze the perfomance of a model is to examine a visualization of its results and they are as follows:

**ResNet50 and EfficientNet-B0**
![resvseff](https://user-images.githubusercontent.com/71333855/130538297-071f99ee-cbe9-400c-b825-6fda295b3232.jpg)

**CNN with No Augmentation**

![noauglossacc](https://user-images.githubusercontent.com/71333855/130538318-a70893d7-7af1-4b7f-ad25-22365dae967c.png)


**5 Block CNN 1 and 2**
![conv2dlossacc](https://user-images.githubusercontent.com/71333855/130538371-4e209c7d-4d78-4a87-9008-aa288b972e45.jpg)


This model's confusion matrix:

![confusionconvd](https://user-images.githubusercontent.com/71333855/130538382-785787a4-b369-40c7-8da6-e62278d0a332.png)

In viewing the matrix, we see 30 representing the fn and the 270 representing the tp from our model, we can deduct that this means that 30 out of 270 skin lesions, or 1 in 9, are misdiagnosed as benign when infact are cancerous. Unfortunately these numbers are not ideal, especially when it comes to the health of patients.

# Conclusion
This project has shown how to benign and malignant diagnosis' from a set of skin lesion images and although it's far from perfect and could be improved, it is amazing to see the success of deep learning being used in real world problems.

# Recommended Next Steps:

** Re-run the models on the full dataset. I've originally done so, however give that the size of the file is so huge (4.18gb, over 25gb with augmentation), I was only able to run one model every few days (each model produced similar results as the ones aboe) and my kernel kept crashing, ultimately leaving me to keep restarting. However, now that I've completed this notebook I am able to simple update the dataset and rerun!

** Re-run some of the models with a greater number of epochs (such as 100 or more on the ones with 30) if necessary to determine if there is convergence. I've attempted this, however my system couldn't handle it.

** Fine tune and test other parameters to reduce overfitting as well as build a model with FN/FP/TN/TP metrics to get a more accurate look when the classes are imbalanced. (18)

** With this project as a base, our work can be built upon to detect more complex problems, such as determining the types of cancers, skin diseases related to Agent Orange and more.

** Output the model to a user friendly application, preferably a web app.

# References

Agent Orange: https://study.com/academy/lesson/what-is-agent-orange-effects-symptoms-definition.html

(1) https://docs.hashicorp.com/sentinel/imports/decimal

(2) https://www.geeksforgeeks.org/convert-bgr-and-rgb-with-python-opencv/

(3) https://datascience.stackexchange.com/questions/41921/sparse-categorical-crossentropy-vs-categorical-crossentropy-keras-accuracy

(4) https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/

(5) https://pythonguides.com/python-print-2-decimal-places/#:~:text=Python%20print%202%20decimal%20places%20In%20Python%2C%20to,will%20print%20the%20float%20with%202%20decimal%20places.

**EfficientNetB0:**

https://keras.io/api/applications/efficientnet/

https://paperswithcode.com/method/efficientnet


(6) https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/l2-regularization

(7) Thank you @python_wizard, @py_lenn, and @synthphreak from reddit for guiding me with this piece of code!

(8) https://keras.io/api/callbacks/

(9) https://keras.io/api/callbacks/reduce_lr_on_plateau/

(10) https://keras.io/api/callbacks/model_checkpoint/

(11) https://keras.io/api/callbacks/early_stopping

(12) https://stackoverflow.com/questions/61362426/why-is-my-val-accuracy-stagnant-at-0-0000e00-while-my-val-loss-is-increasing-fr

(13) https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a

(14) https://stats.stackexchange.com/questions/260505/should-i-use-a-categorical-cross-entropy-or-binary-cross-entropy-loss-for-binary#:~:text=Binary%20cross-entropy%20is%20for%20multi-label%20classifications%2C%20whereas%20categorical,Thanks%20for%20contributing%20an%20answer%20to%20Cross%20Validated%21

(15) https://keras.io/api/layers/convolution_layers/separable_convolution2d/

**ResNet50:**

(16) https://iq.opengenus.org/resnet50-architecture/

(17) https://medium.com/@venkinarayanan/tutorial-image-classifier-using-resnet50-deep-learning-model-python-flask-in-azure-4c2b129af6d2

(18) https://keras.io/examples/structured_data/imbalanced_classification/
