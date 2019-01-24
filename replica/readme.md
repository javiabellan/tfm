# TFM

TFM based on this [paper](https://www.hindawi.com/journals/cin/2018/2061516/)

## Datasets

- [**ISIC2017**](https://challenge.kitware.com/#phase/5840f53ccad3a51cc66c8dab) Dataset of skin lesions. 3 classes, 2000 images:
  - 374 of **malignant** skin tumors: *Melanoma*
  - 1626 of **benign** skin tumors:
    - 254 of *Seborrheic Keratosis*
    - 1372 of *Nevus*

- [**HIS2828**](http://online.unillanos.edu.co:8084/histologyDS/): Dataset of hystology microscopic images. 4 classes, 2828 images:
  - 1026 nervous tissue images
  - 484 connective tissue images
  - 804 epithelial tissue images
  - 514 muscular tissue images


## Models

- SVM (traditional feature)
- **Coding network**: CNN
- **CNMP**: Combines the coding network (CNN) and traditional features **with a Mulilayer Perceptron (automatically)**.
- **R feature fusion**: Combines the coding network (CNN) and traditional features **with a manully fixed proportion**.
- SVM (traditional and deep feature)
- KPCA feature fusion


## 3.1. Coding Network

- Input image: 140×140 RGB (training)
- Normalization: Every image pixel value subtracts the mean RGB value
- Model:
  - Convolutions: 1 pixel stride and 0 pixel padding
  - Max-pooling: with a 5×5 window with a stride of 2 that signifies the overlapping pooling
  - Activation fns: ReLU

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


## 3.2. Traditional Feature Extraction

- **Shape features** (not used)
- **Color histogram features** (not used)
- **Color moment features**: Is based on a single pixel. It is not very sensitive to the angle or size of the image
- **Texture features**: Is a statistical distribution feature that can describe the innate properties of an image surface. It is based on multiple pixel area computing instead of single pixels.

## 3.3. Feature Fusion

2 different fusion approaches to fuse the high-level CNN features (`HF`) and traditional features (`LF`):

#### R feature fusion 

- `feature fusion = λ·LF + (1−λ)·HF`
- λ is the weight parameter that signifies the importance between two different features.
- The feature fusion will feed into softmax to accomplish the last classification task.

#### CNMP

- Train a multilayer perceptron neural network that can fuse the features in nonlinear feature space.




### Data

1. Input image
   - 140×140 RGB image (training)
2. Image preprocessing
   - Subtracting the mean RGB value
   - ZCA whitening
3. Data augmentation
   - extracting random patches from the original image?
   - random cropped to 420×420 and then resized to 140×140
   - random horizontal and vertical? flipping


### CNMP

- Initial weights?
- Dropout, ReLus and batch normalization
- 45 epochs
- Learning rate
  -  To decay the learning rate in each epoch



### Evaluation

- Accuracy: the percentage of correctly classified medical images
- Each dataset was divided into a training set, a validation set, and a test set with the ratio 7:1:2
- All the methods were evaluated using 10-fold cross validation