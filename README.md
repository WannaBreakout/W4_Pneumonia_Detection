# W4_Pneumonia_Detection
Retrain the 'inception v3' image classifier on the pneumonia dataset using Tensorflow+Keras
Jupyter notebook showing the training process and 2 examples of it making predictions on images from the testing dataset. 

## Resource / Reference / Credits to the people who helped directly and indirectly :)


### Dexter1618 - thank you

```
!wget https://data.mendeley.com/datasets/rscbjbr9sj/2/files/41d542e7-7f91-47f6-9ff2-dd8e5a5a7861/ChestXRay2017.zip

from zipfile import ZipFile
with ZipFile("./ChestXRay2017.zip", "r") as f:
    f.extractall(path = "./")

train_files = "./chest_xray/train/"
test_files = "./chest_xray/test/"

```

### vyasmm - nice code - easy to follow
      https://github.com/vyasmm/Pneumonia-Detection



### Anjana Tiha - original example - not clear in many parts

```
Author           : Anjana Tiha
Project Name     : Detection of Pneumonia from Chest X-Ray Images using Convolutional Neural Network, 
                   and Transfer Learning.
Description      : 1. Detected Pneumonia from Chest X-Ray images using Custom Deep Convololutional Neural Network and by retraining pretrained model “InceptionV3” with 5856 images of X-ray (1.15GB).
                 
Tools/Library    : Python, Keras, PyTorch, TensorFlow
Version History  : 1.0.0.0
Current Version  : 1.0.0.3
Last Update      : 12.16.2018

```

### @Stefan999 's suggestion :
```
base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(51024, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
predictions = keras.layers.Dense(2, activation='sigmoid')(x)
model = keras.models.Model(inputs=base_model.inputs, outputs=predictions)
```

### Finally ----> Top Tip  **remove .DS_Store files** - they break things
```
os.remove('./chest_xray/val/.DS_Store')
os.remove('./chest_xray/test/.DS_Store')
os.remove('./chest_xray/train/.DS_Store')
```
