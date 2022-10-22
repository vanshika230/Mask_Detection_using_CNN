# Mask_Detection_using_CNN

CNN stands for Convolutional Neural Network which is a specialized neural network for processing data that has an input shape like a 2D matrix like images. CNN's are typically used for image detection and classification.

Transfer learning is the task of using a pre-trained model and applying it to a new task, i.e., transferring the knowledge learned from one task to another. This is useful because the model doesn’t have to learn from scratch and can achieve higher accuracy in less time as compared to models that don’t use transfer learning.

**The use of transfer learning in the machine learning domain has surged in the last few years. The following are the top reasons:**
- **Growth in the ML community and knowledge sharing**: The research and investments by top universities and tech companies have grown exponentially in the last few years and there is also a strong desire to share state-of-the-art models and datasets with the community. This allows people to utilize pre-trained models in a specific area bootstrap quickly.
- **Common sub-problems**: Another key motivator is that many problems share common sub-problems, e.g., in all visual understanding and prediction areas, tasks such as finding edges, boundaries, and background are common sub-problems. Similarly, in the text domain, the semantic understanding of textual terms can be helpful in almost all problems where the user is represented by text terms, including search, recommendation systems, ads, etc.
- **Limited supervised learning data and training resources**: Many real-world applications are still mapped onto supervised learning problems where the model is asked to predict a label. One key problem is the limited amount of training data available for models to generalize well. One key advantage of doing transfer learning is that we have the ability to start learning from pre-trained models, and hence, we can utilize the knowledge from similar domains.

The transfer learning technique can be utilized in the following ways:
**Extract features from useful layers**
- Keep the initial layers of the pre-trained model and remove the final layers. Add the new layer to the remaining chunk and train them for final classification.
**Fine-tuning**
- Change or tune the existing parameters in a pre-trained network, i.e., optimizing the model parameters during training for the supervised prediction task. A key question with fine-tuning the model is to see how many layers can we freeze and how many final layers we want to fine-tune. This requires understanding the network structure of the model and role of each layer, e.g., for the image classification model we used in the Image data example, once we understand the convolution, pooling, and fully connected layers, we can decide how many final layers we need to fine-tune for our model training process.


#importing libraries
      from fastai import *
      from fastai.vision import *
      from fastai.metrics import error_rate
      import os
      import pandas as pd
      import numpy as np

# Data Loading For training

Things to be remember:

- Decide validation percentage ( 0.2 => 20% )
- Provide path for training data
- Decide augmentations criteria (optional)
- Decide image size (which is 224 in my case)
- Test data can also be added but it's optional
    
    np.random.seed(40)
    data = ImageDataBunch.from_folder(path, train = '.', valid_pct=0.2,
                                  ds_tfms=get_transforms(), size=224,
                                  num_workers=4).normalize(imagenet_stats)
                                  
# Data Explorations

Our image dataset is stored as .jpg files in 2 different folders, with each folder bearing the name of model of the images contained in the folder. We use the ImageDataBunch.from_folder() function to load the images and assign labels the images based on the name of the folder they’re read from.

      data.show_batch(rows=3, figsize=(7,6),recompute_scale_factor=True)
      
# Print Classes present in the data

      data.c — How many classes are there in our dataset?
      len(data.train_ds) — What is the size of our training dataset?
      len(data.valid_ds) — What is the size of our validation dataset?
      print(data.classes)
      len(data.classes)
      data.c
      
 # Create Model


We now use a pre-trained ResNet50 Convolutional Neural Net model, and use transfer learning to learn weights of only the last layer of the network.
Why Transfer learning? Because with transfer learning, you begin with an existing (trained) neural network used for image recognition — and then tweak it a bit (or more) here and there to train a model for your particular use case. And why do we do that? Training a reasonable neural network would mean needing approximately 300,000 image samples, and to achieve really good performance, we’re going to need at least a million images.
In our case, we have approximately 4000+ images in our training set — you have one guess to decide if that would have been enough if were to train a neural net from scratch.
We use the create_cnn() function for loading a pre-trained ResNet18 network, that was trained on around a million images from the ImageNet database.
      learn = cnn_learner(data, models.resnet50, metrics=[accuracy], model_dir = Path('../kaggle/working'),path = Path("."))
  # Finding LR

    learn.lr_find()
    learn.recorder.plot(suggestions=True)
    
   # Train Model

      lr1 = 1e-3
      lr2 = 1e-1
      learn.fit_one_cycle(4,slice(lr1,lr2))
      
      # Hyper Parameter Tuning

      learn.unfreeze()
      learn.fit_one_cycle(10,slice(1e-4,1e-3))
 # Interpret the results

Model performance can be validated in different ways. One of the popular methods is using the confusion matrix. Diagonal values of the matrix indicate correct predictions for each class, whereas other cell values indicate a number of wrong predictions.

      interp = ClassificationInterpretation.from_learner(learn)
      interp.plot_confusion_matrix()
      
      
     # Save and Load Model

Once you have trained the model and satisfied with the outcome, its time to deploy the model. For deploying the model into production you need to save your model architecture and the parameters it’s trained on. For this, the export method is used. The exported model is saved as a PKL file, which is a file created by pickle (a Python module).

        learn.export(file = Path("/kaggle/working/export.pkl"))
        learn.model_dir = "/kaggle/working"
        learn.save("stage-1",return_path=True)
