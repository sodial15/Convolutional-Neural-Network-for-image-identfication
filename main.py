'''
We dont have o type in everything, just understand why
D O   N O T   R U N   O N   L O C A L   M A C H I N E !!!
'''
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# To fix possible crash:
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ================================= LOAD DATASET =================================
'''
We'll be using the CIFAR Image Dataset
60k 32x32 color images
6k images of each class
'''
# Load and split dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be bewteen 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ///////////// VISUALIZATION \\\\\\\\\\\\
IM_INDEX = int(input('Enter image number to visualize: '))
plt.imshow(train_images[IM_INDEX], cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IM_INDEX][0]])
plt.show()

# ================================= CNN BUILDING =================================
# ///////////// CNN ARCHITECTURE \\\\\\\\\\\\
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # Layer 1
model.add(layers.MaxPooling2D((2, 2)))                                            # Layer 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
'''
Layer 1: input shape of data will be 32, 32, 3 and well process 32 filters of 3x3 over input data
         we also apply the relu activation function to every convolution operation output
Layer 2: Will eprform the max pooling operation using 2x2 and stride of 2
Other layers: Similar things,but will take the feature map from previous ones. 
              We also increased frequency form 32 to 64
'''
# Printing a summary
print(model.summary())

'''We have extracted features form the image. Now we need to pass them to a DNN'''

# /// Adding dense Layers \\\
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))  # 10 bc thats the number of classes we have

print()
model.summary()

# ================================= TRAINING =================================
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=4,
                    validation_data=(test_images, test_labels))
# it takes a while to train

# ///// Evaluating \\\\\
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)  # we get arount 67% accuracy

'''
This is a SMALL dataset, even with 60k images (usually trained on millions)
we still need to improve it
Data Augmentation: Improves the model by performing random transformations on our images so that our
model can generalize better (ex: compressions, rotations, stretches, color changes)
It does not overfit the model
'''
# ///// Data Augmentation \\\\
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Create a data generator object that transforms images
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Pick an image to transform
t_img = train_images[14]
img = image.img_to_array(t_img)  # convert image to numpy array
img = img.reshape((1,) + img.shape)  # reshape image

i = 0

for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):  # this loops runs forever until we break
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i > 4:  # show 4 images
        break

    plt.show()

# ////////// Pretraines Modules \\\\\\\\\\
'''
IF even after this we dont have enough images, we can use pretrained modules
COmpanies like google, IBM and such create these open source things
We just fine-tune the last layers of the model so they fit our purposes
'''

import os
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
keras = tf.keras

# ///// Load dataset \\\
''' Well load from tensorflow_datasets
Tis contains image/label pairs where images have different dimensions and 3 color chanels'''

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# split the data manually to divide into 80% training and 10% testing and 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True
)
# Visualization
# Create a function object we can use to get labels
get_label_name = metadata.features['label'].int2str

# Display two images for ds
for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))

# we notice that they are not the same size

# /// Data Preprocessing \\\
IM_SIZE = 160
def format_example(image, label):
    """
    returns an image fthat is reshaped to IM_SIZE
    """
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IM_SIZE, IM_SIZE))
    return image, label

# Now we can appl this function using map
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# Lets have a look at our images now
for image, label in train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))

# Now we will shuffle and batch the images
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_batches = test.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

# Lets compare image sizes now:
for img, label in raw_train.take(2):
    print(f'Original shape: {img.shape}')

for img, label in train.take(2):
    print(f'New shape: {img.shape}')

# /// Pick a pretrained model \\\
# well use MobilNetV2, trained on 1.4M images, developed by Google
# We only want its convolutional base, so well tell it that we dont want the (top)classification base
IMG_SHAPE = (IM_SIZE, IM_SIZE, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# See a summary (incredibly complex design)
base_model.summary()  # Last layer: (None, 5, 5, 1280

'''At this point, the base_model will output a shape of (32, 5, 5, 1280) tensor that is a feature
extraction from our original (1, 160, 160, 3) image.
The 32 means we have 32 layers of filters/features

Because we dont want to train the mdoel again, we have to freeze it
'''
# // Freezing \\
base_model.trainable = False
print(base_model.summary())  # check if we have 0 trainable parameters

# NOW ITS TIME TO ADD OUR CLASSIFIER
# Instead of flattening the base layer well use a global average polling layer that will
# averag the entire 5x5 area of each 2D feature map and return us a single 1280 element vector per file
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# Finally we will add the prdiction layer, that will be a single dense neuron.
# We can do this because we only have two classes to predict for
prediction_layer = keras.layers.Dense(1)

# Combine layers in a model
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

# Print a summary
print(model.summary())

# ///// Traing the model \\\\\
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Evaluate the model
# We can evaluate the model now, to see how it does before passing it our new images
initial_epochs = 3
validation_steps = 20

loss0, accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

# Were getting accuracy of 56%, its practivally guessing
# =================== TRAIN THE MODEL ON OUR IMAGES ======================
# (takes like 30 minutes)
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

acc = history.history['accuracy']
print(acc)

# Save the model
model.save('dogs_vs_cats.h5')  # h5 is a keras format for models
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')

# Predict
# model.predict()
