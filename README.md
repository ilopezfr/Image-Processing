# Image Processing Master Repo

This repo is intended to serve as a resource for any type of work related with Computer Vision and Image Processing (i.e.: Object Detection, Segmentation, Classification, etc.). For me is a way to keep everything I learn in a central place. Therefore, it's a dynamic repo in the sense that I'll keep adding and orginizing resources as I continue learning and experimenting in this field. 

## Table of Contents

- [Dataset](#dataset)
    - 
- [Computational Photography](#computational-photography)
- [Object Recognition](#Object-Recognition)
    - 
    - 
- [Facial Recognition](#google-colab)
- [Image Understanding](#image-understanding)
  - [Semantic Segmentation](#semantic-segmentation)]
  -  
- [Video Analytics](#video-processing)



Dataset
---










Object Recognition
---


### Best Practices: 
- Training data should be as close as possible to the data feeded to the model in the deployment phase. If you plan to deploy your model in a low-res camera, you should include blurry, low-res images on your train data. 
- For training, it's recommended to use at least 50 images per label. Although keep in mind that the more used, the better the predictions.
- The model we will be training resizes the image to 300x300 pixels, so keep that is mind when training the model with images where one dimension is much longer than the other. 

