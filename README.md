# cat_dog

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.applications import MobileNetV2

!mkdir -p ~/.kaggle
!cp kaggle-2.json ~/.kaggle/

cp: cannot stat 'kaggle-2.json': No such file or directory

[ ]
! kaggle datasets download -d "ashfakyeafi/cat-dog-images-for-classification"
Dataset URL: https://www.kaggle.com/datasets/ashfakyeafi/cat-dog-images-for-classification
License(s): CC0-1.0
Downloading cat-dog-images-for-classification.zip to /content
100% 544M/545M [00:15<00:00, 41.4MB/s]
100% 545M/545M [00:15<00:00, 38.0MB/s]

[ ]
import zipfile
zip_ref = zipfile.ZipFile('/content/cat-dog-images-for-classification.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

[ ]
data=pd.read_csv('/content/cat_dog.csv')

[ ]
data.head()


[ ]
# Add the full image path to the DataFrame
data["image_path"] = data["image"].apply(lambda x: os.path.join("/content/cat_dog", x))

# Filter out files that are not present in the directory
data = data[data["image_path"].apply(os.path.exists)]

# Split the data into training, validation, and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, stratify=data["labels"], random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, stratify=train_data["labels"], random_state=42)


[ ]
# Data generators for memory efficiency
image_size = (128, 128)
batch_size = 32

# Data Augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


[ ]
def preprocess_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = img / 255.0
    return img, label


[ ]
# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_data["image_path"].values, train_data["labels"].values))
train_dataset = train_dataset.map(preprocess_image).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_data["image_path"].values, val_data["labels"].values))
val_dataset = val_dataset.map(preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_data["image_path"].values, test_data["labels"].values))
test_dataset = test_dataset.map(preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)



[ ]
# Build a CNN model
model1 = Sequential([
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

[ ]
# Compile the model with a reduced learning rate
optimizer = Adam(learning_rate=0.0001)
model1.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks for Early Stopping and Learning Rate Reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# Train the model
history = model1.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)


Epoch 1/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 36s 51ms/step - accuracy: 0.6403 - loss: 4.2598 - val_accuracy: 0.6440 - val_loss: 3.3514 - learning_rate: 1.0000e-04
Epoch 2/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 33s 49ms/step - accuracy: 0.7632 - loss: 2.8814 - val_accuracy: 0.7832 - val_loss: 2.2460 - learning_rate: 1.0000e-04
Epoch 3/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 41s 50ms/step - accuracy: 0.8053 - loss: 2.0552 - val_accuracy: 0.7878 - val_loss: 1.7421 - learning_rate: 1.0000e-04
Epoch 4/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 28s 53ms/step - accuracy: 0.8327 - loss: 1.5596 - val_accuracy: 0.8223 - val_loss: 1.3667 - learning_rate: 1.0000e-04
Epoch 5/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 37s 46ms/step - accuracy: 0.8637 - loss: 1.2449 - val_accuracy: 0.8305 - val_loss: 1.1758 - learning_rate: 1.0000e-04
Epoch 6/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 42s 46ms/step - accuracy: 0.8855 - loss: 1.0536 - val_accuracy: 0.8295 - val_loss: 1.1114 - learning_rate: 1.0000e-04
Epoch 7/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 31s 59ms/step - accuracy: 0.8994 - loss: 0.9349 - val_accuracy: 0.8292 - val_loss: 1.0411 - learning_rate: 1.0000e-04
Epoch 8/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 26s 50ms/step - accuracy: 0.9240 - loss: 0.8319 - val_accuracy: 0.8425 - val_loss: 1.0104 - learning_rate: 1.0000e-04
Epoch 9/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 39s 46ms/step - accuracy: 0.9450 - loss: 0.7540 - val_accuracy: 0.8290 - val_loss: 0.9870 - learning_rate: 1.0000e-04
Epoch 10/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 43s 49ms/step - accuracy: 0.9505 - loss: 0.7059 - val_accuracy: 0.8455 - val_loss: 0.9651 - learning_rate: 1.0000e-04
Epoch 11/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 24s 46ms/step - accuracy: 0.9648 - loss: 0.6529 - val_accuracy: 0.8460 - val_loss: 0.9281 - learning_rate: 1.0000e-04
Epoch 12/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 26s 49ms/step - accuracy: 0.9690 - loss: 0.6204 - val_accuracy: 0.8430 - val_loss: 0.9184 - learning_rate: 1.0000e-04
Epoch 13/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 39s 47ms/step - accuracy: 0.9739 - loss: 0.5841 - val_accuracy: 0.8415 - val_loss: 0.9458 - learning_rate: 1.0000e-04
Epoch 14/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 40s 46ms/step - accuracy: 0.9748 - loss: 0.5678 - val_accuracy: 0.8432 - val_loss: 0.9155 - learning_rate: 1.0000e-04
Epoch 15/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 26s 51ms/step - accuracy: 0.9771 - loss: 0.5507 - val_accuracy: 0.8328 - val_loss: 1.0016 - learning_rate: 1.0000e-04
Epoch 16/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 40s 49ms/step - accuracy: 0.9749 - loss: 0.5425 - val_accuracy: 0.8105 - val_loss: 1.0399 - learning_rate: 1.0000e-04
Epoch 17/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 0s 43ms/step - accuracy: 0.9758 - loss: 0.5336
Epoch 17: ReduceLROnPlateau reducing learning rate to 4.999999873689376e-05.
500/500 ━━━━━━━━━━━━━━━━━━━━ 26s 50ms/step - accuracy: 0.9758 - loss: 0.5336 - val_accuracy: 0.8487 - val_loss: 0.9542 - learning_rate: 1.0000e-04
Epoch 18/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 27s 52ms/step - accuracy: 0.9862 - loss: 0.5038 - val_accuracy: 0.8570 - val_loss: 0.8826 - learning_rate: 5.0000e-05
Epoch 19/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 40s 49ms/step - accuracy: 0.9956 - loss: 0.4275 - val_accuracy: 0.8547 - val_loss: 0.8376 - learning_rate: 5.0000e-05
Epoch 20/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 26s 51ms/step - accuracy: 0.9965 - loss: 0.3786 - val_accuracy: 0.8472 - val_loss: 0.8506 - learning_rate: 5.0000e-05
Epoch 21/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 24s 46ms/step - accuracy: 0.9958 - loss: 0.3448 - val_accuracy: 0.8547 - val_loss: 0.8131 - learning_rate: 5.0000e-05
Epoch 22/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 45s 53ms/step - accuracy: 0.9957 - loss: 0.3261 - val_accuracy: 0.8490 - val_loss: 0.7681 - learning_rate: 5.0000e-05
Epoch 23/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 26s 50ms/step - accuracy: 0.9916 - loss: 0.3246 - val_accuracy: 0.8585 - val_loss: 0.7785 - learning_rate: 5.0000e-05
Epoch 24/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 24s 46ms/step - accuracy: 0.9923 - loss: 0.3175 - val_accuracy: 0.8555 - val_loss: 0.7878 - learning_rate: 5.0000e-05
Epoch 25/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 0s 43ms/step - accuracy: 0.9919 - loss: 0.3151
Epoch 25: ReduceLROnPlateau reducing learning rate to 2.499999936844688e-05.
500/500 ━━━━━━━━━━━━━━━━━━━━ 43s 52ms/step - accuracy: 0.9919 - loss: 0.3151 - val_accuracy: 0.8522 - val_loss: 0.8429 - learning_rate: 5.0000e-05
Epoch 26/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 26s 50ms/step - accuracy: 0.9956 - loss: 0.2966 - val_accuracy: 0.8612 - val_loss: 0.7663 - learning_rate: 2.5000e-05
Epoch 27/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 40s 47ms/step - accuracy: 0.9998 - loss: 0.2689 - val_accuracy: 0.8620 - val_loss: 0.7667 - learning_rate: 2.5000e-05
Epoch 28/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 27s 52ms/step - accuracy: 0.9993 - loss: 0.2503 - val_accuracy: 0.8560 - val_loss: 0.7170 - learning_rate: 2.5000e-05
Epoch 29/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 24s 47ms/step - accuracy: 0.9992 - loss: 0.2341 - val_accuracy: 0.8587 - val_loss: 0.7166 - learning_rate: 2.5000e-05
Epoch 30/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 46s 55ms/step - accuracy: 0.9995 - loss: 0.2182 - val_accuracy: 0.8568 - val_loss: 0.7439 - learning_rate: 2.5000e-05
Epoch 31/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 26s 51ms/step - accuracy: 0.9989 - loss: 0.2107 - val_accuracy: 0.8545 - val_loss: 0.7225 - learning_rate: 2.5000e-05
Epoch 32/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 39s 46ms/step - accuracy: 0.9973 - loss: 0.2121 - val_accuracy: 0.8580 - val_loss: 0.6942 - learning_rate: 2.5000e-05
Epoch 33/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 43s 51ms/step - accuracy: 0.9986 - loss: 0.2043 - val_accuracy: 0.8618 - val_loss: 0.7142 - learning_rate: 2.5000e-05
Epoch 34/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 24s 47ms/step - accuracy: 0.9989 - loss: 0.1957 - val_accuracy: 0.8675 - val_loss: 0.6854 - learning_rate: 2.5000e-05
Epoch 35/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 27s 51ms/step - accuracy: 0.9998 - loss: 0.1859 - val_accuracy: 0.8593 - val_loss: 0.7348 - learning_rate: 2.5000e-05
Epoch 36/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 40s 50ms/step - accuracy: 0.9979 - loss: 0.1863 - val_accuracy: 0.8575 - val_loss: 0.7059 - learning_rate: 2.5000e-05
Epoch 37/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 41s 50ms/step - accuracy: 0.9986 - loss: 0.1812 - val_accuracy: 0.8608 - val_loss: 0.6660 - learning_rate: 2.5000e-05
Epoch 38/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 27s 52ms/step - accuracy: 0.9986 - loss: 0.1810 - val_accuracy: 0.8577 - val_loss: 0.6858 - learning_rate: 2.5000e-05
Epoch 39/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 24s 47ms/step - accuracy: 0.9984 - loss: 0.1773 - val_accuracy: 0.8645 - val_loss: 0.6637 - learning_rate: 2.5000e-05
Epoch 40/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 27s 52ms/step - accuracy: 0.9973 - loss: 0.1761 - val_accuracy: 0.8550 - val_loss: 0.6749 - learning_rate: 2.5000e-05
Epoch 41/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 24s 46ms/step - accuracy: 0.9988 - loss: 0.1708 - val_accuracy: 0.8550 - val_loss: 0.6602 - learning_rate: 2.5000e-05
Epoch 42/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 26s 51ms/step - accuracy: 0.9992 - loss: 0.1650 - val_accuracy: 0.8590 - val_loss: 0.6729 - learning_rate: 2.5000e-05
Epoch 43/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 39s 47ms/step - accuracy: 0.9989 - loss: 0.1616 - val_accuracy: 0.8560 - val_loss: 0.6624 - learning_rate: 2.5000e-05
Epoch 44/100
499/500 ━━━━━━━━━━━━━━━━━━━━ 0s 43ms/step - accuracy: 0.9996 - loss: 0.1556
Epoch 44: ReduceLROnPlateau reducing learning rate to 1.249999968422344e-05.
500/500 ━━━━━━━━━━━━━━━━━━━━ 28s 53ms/step - accuracy: 0.9996 - loss: 0.1556 - val_accuracy: 0.8457 - val_loss: 0.7305 - learning_rate: 2.5000e-05
Epoch 45/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 40s 53ms/step - accuracy: 0.9990 - loss: 0.1544 - val_accuracy: 0.8658 - val_loss: 0.6296 - learning_rate: 1.2500e-05
Epoch 46/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 26s 50ms/step - accuracy: 1.0000 - loss: 0.1458 - val_accuracy: 0.8720 - val_loss: 0.6231 - learning_rate: 1.2500e-05
Epoch 47/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 41s 51ms/step - accuracy: 1.0000 - loss: 0.1402 - val_accuracy: 0.8687 - val_loss: 0.6010 - learning_rate: 1.2500e-05
Epoch 48/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 24s 46ms/step - accuracy: 1.0000 - loss: 0.1349 - val_accuracy: 0.8735 - val_loss: 0.5708 - learning_rate: 1.2500e-05
Epoch 49/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 31s 61ms/step - accuracy: 0.9999 - loss: 0.1305 - val_accuracy: 0.8733 - val_loss: 0.5762 - learning_rate: 1.2500e-05
Epoch 50/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 34s 47ms/step - accuracy: 0.9999 - loss: 0.1270 - val_accuracy: 0.8705 - val_loss: 0.5627 - learning_rate: 1.2500e-05
Epoch 51/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 43s 50ms/step - accuracy: 1.0000 - loss: 0.1235 - val_accuracy: 0.8725 - val_loss: 0.5520 - learning_rate: 1.2500e-05
Epoch 52/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 40s 51ms/step - accuracy: 0.9999 - loss: 0.1196 - val_accuracy: 0.8725 - val_loss: 0.5380 - learning_rate: 1.2500e-05
Epoch 53/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 39s 46ms/step - accuracy: 1.0000 - loss: 0.1172 - val_accuracy: 0.8618 - val_loss: 0.5448 - learning_rate: 1.2500e-05
Epoch 54/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 45s 54ms/step - accuracy: 0.9997 - loss: 0.1169 - val_accuracy: 0.8652 - val_loss: 0.5236 - learning_rate: 1.2500e-05
Epoch 55/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 26s 50ms/step - accuracy: 0.9999 - loss: 0.1136 - val_accuracy: 0.8700 - val_loss: 0.5501 - learning_rate: 1.2500e-05
Epoch 56/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 40s 47ms/step - accuracy: 0.9999 - loss: 0.1121 - val_accuracy: 0.8665 - val_loss: 0.5403 - learning_rate: 1.2500e-05
Epoch 57/100
499/500 ━━━━━━━━━━━━━━━━━━━━ 0s 49ms/step - accuracy: 1.0000 - loss: 0.1093
Epoch 57: ReduceLROnPlateau reducing learning rate to 6.24999984211172e-06.
500/500 ━━━━━━━━━━━━━━━━━━━━ 45s 57ms/step - accuracy: 1.0000 - loss: 0.1093 - val_accuracy: 0.8687 - val_loss: 0.5337 - learning_rate: 1.2500e-05
Epoch 58/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 35s 46ms/step - accuracy: 1.0000 - loss: 0.1079 - val_accuracy: 0.8758 - val_loss: 0.5325 - learning_rate: 6.2500e-06
Epoch 59/100
500/500 ━━━━━━━━━━━━━━━━━━━━ 43s 49ms/step - accuracy: 1.0000 - loss: 0.1052 - val_accuracy: 0.8695 - val_loss: 0.5408 - learning_rate: 6.2500e-06
Epoch 59: early stopping
Restoring model weights from the end of the best epoch: 54.

[ ]
model1.summary()



[ ]
# Evaluate the model on the test dataset
test_loss, test_accuracy = model1.evaluate(test_dataset, verbose=1)
print(f"Test Accuracy: {test_accuracy:.2f}, Test Loss: {test_loss:.2f}")
157/157 ━━━━━━━━━━━━━━━━━━━━ 6s 35ms/step - accuracy: 0.8760 - loss: 0.5333
Test Accuracy: 0.88, Test Loss: 0.53

[ ]
# Sample 50 images from the test dataset
sample_dataset = test_dataset.unbatch().take(100)

# Prepare lists to store images, true labels, and predictions
images, true_labels, predictions = [], [], []

# Collect the images, true labels, and predictions
for img, label in sample_dataset:
    images.append(img.numpy())
    true_labels.append(label.numpy())
    prediction = model1.predict(tf.expand_dims(img, axis=0))[0][0]
    predictions.append(prediction)

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 51ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step

[ ]
# Plot the 10x10 grid
fig, axes = plt.subplots(10, 10, figsize=(15, 15))
axes = axes.flatten()

for i in range(100):
    ax = axes[i]
    ax.imshow(images[i])
    ax.axis('off')

    true_label = "Dog" if true_labels[i] == 1 else "Cat"
    predicted_label = "Dog" if predictions[i] > 0.5 else "Cat"
    ax.set_title(f"Actual: {true_label}\nPredicted: {predicted_label}", fontsize=8)

plt.tight_layout()
plt.show()



[ ]

# Use MobileNetV2 for transfer learning
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Build the model
model2 = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),
    Dropout(0.6),
    Dense(1, activation='sigmoid')
])

# Compile the model
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Reduced learning rate
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Train the model
history = model2.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[early_stopping, reduce_lr]
)




Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5
9406464/9406464 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step
Epoch 1/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 44s 66ms/step - accuracy: 0.9252 - loss: 3.7316 - val_accuracy: 0.9672 - val_loss: 1.5776 - learning_rate: 1.0000e-04
Epoch 2/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 26s 48ms/step - accuracy: 0.9651 - loss: 1.3095 - val_accuracy: 0.9645 - val_loss: 0.7290 - learning_rate: 1.0000e-04
Epoch 3/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 30s 57ms/step - accuracy: 0.9661 - loss: 0.6279 - val_accuracy: 0.9678 - val_loss: 0.4087 - learning_rate: 1.0000e-04
Epoch 4/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 38s 52ms/step - accuracy: 0.9690 - loss: 0.3677 - val_accuracy: 0.9680 - val_loss: 0.2836 - learning_rate: 1.0000e-04
Epoch 5/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 39s 47ms/step - accuracy: 0.9707 - loss: 0.2609 - val_accuracy: 0.9635 - val_loss: 0.2430 - learning_rate: 1.0000e-04
Epoch 6/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 32s 62ms/step - accuracy: 0.9667 - loss: 0.2346 - val_accuracy: 0.9680 - val_loss: 0.2088 - learning_rate: 1.0000e-04
Epoch 7/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 31s 61ms/step - accuracy: 0.9697 - loss: 0.2001 - val_accuracy: 0.9582 - val_loss: 0.2366 - learning_rate: 1.0000e-04
Epoch 8/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 26s 51ms/step - accuracy: 0.9686 - loss: 0.1969 - val_accuracy: 0.9660 - val_loss: 0.1931 - learning_rate: 1.0000e-04
Epoch 9/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 40s 49ms/step - accuracy: 0.9709 - loss: 0.1788 - val_accuracy: 0.9665 - val_loss: 0.1824 - learning_rate: 1.0000e-04
Epoch 10/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 43s 54ms/step - accuracy: 0.9704 - loss: 0.1766 - val_accuracy: 0.9665 - val_loss: 0.1827 - learning_rate: 1.0000e-04
Epoch 11/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 38s 48ms/step - accuracy: 0.9704 - loss: 0.1764 - val_accuracy: 0.9670 - val_loss: 0.1792 - learning_rate: 1.0000e-04
Epoch 12/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 29s 54ms/step - accuracy: 0.9707 - loss: 0.1764 - val_accuracy: 0.9707 - val_loss: 0.1766 - learning_rate: 1.0000e-04
Epoch 13/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 27s 52ms/step - accuracy: 0.9689 - loss: 0.1746 - val_accuracy: 0.9663 - val_loss: 0.1786 - learning_rate: 1.0000e-04
Epoch 14/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 25s 48ms/step - accuracy: 0.9705 - loss: 0.1679 - val_accuracy: 0.9635 - val_loss: 0.1831 - learning_rate: 1.0000e-04
Epoch 15/50
499/500 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - accuracy: 0.9706 - loss: 0.1655
Epoch 15: ReduceLROnPlateau reducing learning rate to 4.999999873689376e-05.
500/500 ━━━━━━━━━━━━━━━━━━━━ 43s 54ms/step - accuracy: 0.9706 - loss: 0.1655 - val_accuracy: 0.9628 - val_loss: 0.1850 - learning_rate: 1.0000e-04
Epoch 16/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 39s 51ms/step - accuracy: 0.9757 - loss: 0.1493 - val_accuracy: 0.9682 - val_loss: 0.1499 - learning_rate: 5.0000e-05
Epoch 17/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 28s 54ms/step - accuracy: 0.9804 - loss: 0.1180 - val_accuracy: 0.9680 - val_loss: 0.1408 - learning_rate: 5.0000e-05
Epoch 18/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 44s 61ms/step - accuracy: 0.9816 - loss: 0.1044 - val_accuracy: 0.9678 - val_loss: 0.1373 - learning_rate: 5.0000e-05
Epoch 19/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 25s 49ms/step - accuracy: 0.9802 - loss: 0.1039 - val_accuracy: 0.9680 - val_loss: 0.1325 - learning_rate: 5.0000e-05
Epoch 20/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 44s 53ms/step - accuracy: 0.9820 - loss: 0.1007 - val_accuracy: 0.9647 - val_loss: 0.1393 - learning_rate: 5.0000e-05
Epoch 21/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 30s 58ms/step - accuracy: 0.9837 - loss: 0.1002 - val_accuracy: 0.9697 - val_loss: 0.1292 - learning_rate: 5.0000e-05
Epoch 22/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 26s 51ms/step - accuracy: 0.9824 - loss: 0.0976 - val_accuracy: 0.9685 - val_loss: 0.1353 - learning_rate: 5.0000e-05
Epoch 23/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 41s 50ms/step - accuracy: 0.9834 - loss: 0.0990 - val_accuracy: 0.9685 - val_loss: 0.1357 - learning_rate: 5.0000e-05
Epoch 24/50
498/500 ━━━━━━━━━━━━━━━━━━━━ 0s 43ms/step - accuracy: 0.9826 - loss: 0.0980
Epoch 24: ReduceLROnPlateau reducing learning rate to 2.499999936844688e-05.
500/500 ━━━━━━━━━━━━━━━━━━━━ 27s 53ms/step - accuracy: 0.9826 - loss: 0.0980 - val_accuracy: 0.9678 - val_loss: 0.1348 - learning_rate: 5.0000e-05
Epoch 25/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 26s 51ms/step - accuracy: 0.9864 - loss: 0.0877 - val_accuracy: 0.9685 - val_loss: 0.1283 - learning_rate: 2.5000e-05
Epoch 26/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 27s 51ms/step - accuracy: 0.9893 - loss: 0.0765 - val_accuracy: 0.9697 - val_loss: 0.1235 - learning_rate: 2.5000e-05
Epoch 27/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 45s 61ms/step - accuracy: 0.9911 - loss: 0.0703 - val_accuracy: 0.9695 - val_loss: 0.1203 - learning_rate: 2.5000e-05
Epoch 28/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 35s 48ms/step - accuracy: 0.9912 - loss: 0.0653 - val_accuracy: 0.9690 - val_loss: 0.1203 - learning_rate: 2.5000e-05
Epoch 29/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 45s 55ms/step - accuracy: 0.9911 - loss: 0.0636 - val_accuracy: 0.9697 - val_loss: 0.1213 - learning_rate: 2.5000e-05
Epoch 30/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 44s 62ms/step - accuracy: 0.9912 - loss: 0.0619 - val_accuracy: 0.9690 - val_loss: 0.1184 - learning_rate: 2.5000e-05
Epoch 31/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 36s 51ms/step - accuracy: 0.9912 - loss: 0.0616 - val_accuracy: 0.9697 - val_loss: 0.1168 - learning_rate: 2.5000e-05
Epoch 32/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 42s 55ms/step - accuracy: 0.9910 - loss: 0.0597 - val_accuracy: 0.9697 - val_loss: 0.1160 - learning_rate: 2.5000e-05
Epoch 33/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 27s 52ms/step - accuracy: 0.9911 - loss: 0.0592 - val_accuracy: 0.9693 - val_loss: 0.1208 - learning_rate: 2.5000e-05
Epoch 34/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 43s 53ms/step - accuracy: 0.9907 - loss: 0.0609 - val_accuracy: 0.9697 - val_loss: 0.1147 - learning_rate: 2.5000e-05
Epoch 35/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 40s 52ms/step - accuracy: 0.9921 - loss: 0.0596 - val_accuracy: 0.9693 - val_loss: 0.1240 - learning_rate: 2.5000e-05
Epoch 36/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 39s 48ms/step - accuracy: 0.9917 - loss: 0.0581 - val_accuracy: 0.9693 - val_loss: 0.1205 - learning_rate: 2.5000e-05
Epoch 37/50
499/500 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step - accuracy: 0.9910 - loss: 0.0574
Epoch 37: ReduceLROnPlateau reducing learning rate to 1.249999968422344e-05.
500/500 ━━━━━━━━━━━━━━━━━━━━ 31s 61ms/step - accuracy: 0.9910 - loss: 0.0574 - val_accuracy: 0.9672 - val_loss: 0.1215 - learning_rate: 2.5000e-05
Epoch 38/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 32s 63ms/step - accuracy: 0.9948 - loss: 0.0521 - val_accuracy: 0.9680 - val_loss: 0.1221 - learning_rate: 1.2500e-05
Epoch 39/50
500/500 ━━━━━━━━━━━━━━━━━━━━ 36s 52ms/step - accuracy: 0.9952 - loss: 0.0477 - val_accuracy: 0.9705 - val_loss: 0.1150 - learning_rate: 1.2500e-05
Epoch 39: early stopping
Restoring model weights from the end of the best epoch: 34.

[ ]
model2.summary()



[ ]

# Evaluate the model on the test dataset
test_loss, test_accuracy = model2.evaluate(test_dataset, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%, Test Loss: {test_loss:.4f}")
157/157 ━━━━━━━━━━━━━━━━━━━━ 11s 68ms/step - accuracy: 0.9680 - loss: 0.1175
Test Accuracy: 97.12%, Test Loss: 0.1129

[ ]
# Plot training and validation curves
def plot_learning_curves(history):
    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.show()

plot_learning_curves(history)


[ ]
# Sample 100 images from the test dataset
sample_dataset = test_dataset.unbatch().take(100)

# Prepare lists to store images, true labels, and predictions
images, true_labels, predictions = [], [], []

# Collect the images, true labels, and predictions
for img, label in sample_dataset:
    images.append(img.numpy())
    true_labels.append(label.numpy())
    prediction = model2.predict(tf.expand_dims(img, axis=0))[0][0]
    predictions.append(prediction)


1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 48ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 51ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 51ms/step

[ ]
# Plot the 10x10 grid
fig, axes = plt.subplots(10, 10, figsize=(15, 15))
axes = axes.flatten()

for i in range(100):
    ax = axes[i]
    ax.imshow(images[i])
    ax.axis('off')

    true_label = "Dog" if true_labels[i] == 1 else "Cat"
    predicted_label = "Dog" if predictions[i] > 0.5 else "Cat"
    ax.set_title(f"Actual: {true_label}\nPredicted: {predicted_label}", fontsize=8)

plt.tight_layout()
plt.show()



[ ]
import tensorflow as tf
import matplotlib.pyplot as plt

def preprocess_single_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (128, 128))
    img = img / 255.0
    return tf.expand_dims(img, axis=0)

def predict_and_display(image_path, model):
    processed_img = preprocess_single_image(image_path)

    prediction = model.predict(processed_img)[0][0]
    predicted_label = "Dog" if prediction > 0.5 else "Cat"

    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    plt.imshow(img.numpy())
    plt.axis('off')
    plt.title(f"Predicted: {predicted_label}")
    plt.show()


[ ]
image_paths = [
    "/content/cat1.jpg",
    "/content/cat2.jpg",
    "/content/dog1.jpg",
    "/content/dog2.jpg",
    "/content/dog3.jpg",
    "/content/cat3.jpg"

]

[ ]
# using model1 to predict cat or dog

for img_path in image_paths:
    predict_and_display(img_path, model1)



[ ]
# using model2 to predict cat or dog

for img_path in image_paths:
    predict_and_display(img_path, model2)


