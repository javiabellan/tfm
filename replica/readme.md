# TFM

TFM based on this [paper](https://www.hindawi.com/journals/cin/2018/2061516/)

## Datasets

- [**ISIC2017**](https://challenge.kitware.com/#phase/5840f53ccad3a51cc66c8dab) Dataset of skin lesions. 2 binary image classification tasks. 2000 images, variable image sizes:
  - 374 of **malignant** skin tumors: *Melanoma*
  - 1626 of **benign** skin tumors:
    - 254 of *Seborrheic Keratosis*
    - 1372 of *Nevus*

- [**HIS2828**](http://online.unillanos.edu.co:8084/histologyDS/): Dataset of hystology microscopic images. Classification task with 4 classes. 2828 images, 720×480 image size:
  - 1026 nervous tissue images
  - 484 connective tissue images
  - 804 epithelial tissue images
  - 514 muscular tissue images

> - Each dataset was divided into a training set, a validation set, and a test set with the ratio 7:1:2
> - All the alorithims were evaluated using 10-fold cross validation.

## Traditional Feature Extraction

- Implemented in MatConvn (matlab toolbox)
- **Shape features** (not used)
- **Color histogram features** (not used)
- **Color moment features**: Is based on a single pixel. It is not very sensitive to the angle or size of the image
- **Texture features**: Is a statistical distribution feature that can describe the innate properties of an image surface. It is based on multiple pixel area computing instead of single pixels.


## Models

Ordered from best to worse:

1. **CNMP**: Combines the coding network (CNN) and traditional features **with a Multilayer Perceptron (automatically)**.
2. **R feature fusion**: Combines the coding network (CNN) and traditional features **with a manully fixed proportion**.
3. **KPCA feature fusion**
4. **SVM (traditional and deep feature)**
5. **Coding network**: CNN
6. **SVM (traditional feature)**



## 1. Coding Network

- Implemented in MatConvn (matlab toolbox)
- Input image: Size of 140×140 (training)
  - For HIS2828 dataset: Randomly cropped to 420×420, and then resized to 140×140.
  - For ISIC2017 dataset: Randomly cropped with two-thirds of the original height and width, and then resized to 140×140.
- Augmentation: Flip the image horizontality or verticality.
- Normalization:
  - Every image pixel value subtracts the mean RGB value
  - ZCA whitening??
- Model:
  - Convolutions: 1 pixel stride and 0 pixel padding
  - Max-pooling: with a 5×5 window with a stride of 2 that signifies the overlapping pooling
  - Activation fns: ReLU
  - Batch normalization: Used
  - Dropout: Used
- Train
  - 45 epochs
  - Random initial weights
  - Learning rate: To decay the learning rate in each epoch
- Test
  - At test time, the network makes a prediction to each patche and average the predictions.

Layer       | Kernel | Stride | Output size
------------|--------|--------|------------
Convolution | 11×11  |    1   | 130×130×32
Convolution | 11×11  |    1   | 120×120×32
Max pooling | 5×5    |    2   | 58×58×32
Convolution | 9×9    |    1   | 50×50×64
Max pooling | 5×5    |    2   | 23×23×64
Convolution | 8×8    |    1   | 16×16×128
Convolution | 9×9    |    1   | 8×8×256
Convolution (high-level features) | 8×8    |    1   | 1×1×256
Dense       |    -   |    -   | 1×1×4
Softmax     |    -   |    -   | 1×1×4



## 2. R feature fusion

- Manually fuse high-level CNN features (`HF`) with traditional features (`LF`).
- `feature fusion = λ·LF + (1−λ)·HF`
- `λ` is the weight parameter that signifies the importance between two different features.
- The feature fusion will feed into softmax to accomplish the last classification task.


## 3. CNMP feature fusion

- Train a multilayer perceptron neural network that can fuse the features in nonlinear feature space.


## 4. KPCA feature fusion

- With the RBF kernal to fuse features is that it could map the feature into nonlinear space as well.
- The feature fusion vector will feed into softmax to finish classification task.


## 5. SVM (traditional features)

- Implemented in LibSVM-3.17 package.
- Trainned a one-vs-one multiclass classifier with radial basis function (RBF) kernal.
- It uses little image information along with abandoning a large amount of the image’s spatial information.
- This model must train very few parameters with respect to the deep model.