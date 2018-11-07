# TFM

TFM based on this [paper](https://www.hindawi.com/journals/cin/2018/2061516/)

### Datasets

- [**ISIC2017**](https://challenge.kitware.com/#phase/5840f53ccad3a51cc66c8dab) Datasets of skin lesions and skin cancer. 2000 images, format: JPG
  - 374 of **malignant** skin tumors: *Melanoma*
  - 1626 of **benign** skin tumors:
    - 254 of *Seborrheic Keratosis*
    - 1372 of *Nevus*

- [**HIS2828**](http://online.unillanos.edu.co:8084/histologyDS/): Datasets of hystology microscopic images. 4 classes of tissue. 2828 images of 720x480
  - 1026 nervous tissue images
  - 484 connective tissue images
  - 804 epithelial tissue images
  - 514 muscular tissue images


### Data

1. Input image
   - RGB hisology image
2. Image preprocessing
   - Subtracting the mean RGB value
   - ZCA whitening
3. Data augmentation
   - random cropped to 420×420 and then resized to 140×140
   - random horizontal and vertical flipping


### Model

- initial weights
- dropout, ReLus and batch normalization
- 45 epochs
- learning rate
  -  To decay the learning rate in each epoch

  
Layer       | Kernel | Stride | Output size
------------|--------|--------|------------
Convolution | 11×11  |    1   | 130×130×32
Convolution | 11×11  |    1   | 120×120×32
Max pooling | 5×5    |    2   | 58×58×32
Convolution | 9×9    |    1   | 50×50×64
Max pooling | 5×5    |    2   | 23×23×64
Convolution | 8×8    |    1   | 16×16×128
Convolution | 9×9    |    1   | 8×8×256
Convolution | 8×8    |    1   | 1×1×256
Rasterize   |    -   |    -   | 1×1×4
Softmax     |    -   |    -   | 1×1×4

### Evaluation

- Accuracy: the percentage of correctly classified medical images
- Each dataset was divided into a training set, a validation set, and a test set with the ratio 7:1:2
- All the methods were evaluated using 10-fold cross validation