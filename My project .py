#!/usr/bin/env python
# coding: utf-8

# In[27]:


import zipfile

with zipfile.ZipFile('project2.zip', 'r') as zip_ref:
    zip_ref.extractall()


# In[28]:


import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
print(tf.__version__)
############ settings ############
data_dir = r"..\data"
batch_size = 32
img_height = 224
img_width = 224


# In[29]:


import tensorflow as tf
# Replace this with the actual path to your dataset
data_dir = "C:/Users/HOME/Desktop/project2"
# Replace these with the desired image dimensions and batch size
img_height = 128
img_width = 128
batch_size = 32
# Create the training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
data_dir,
validation_split=0.3,
subset="training",
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size)
# Create the validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
data_dir,
validation_split=0.3,
subset="validation",
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size)


# In[26]:


import matplotlib.pyplot as plt
from collections import Counter

# Extract labels from training dataset
train_labels = [label for _, label in train_ds.unbatch()]
train_labels = [label.numpy() for label in train_labels]

# Extract labels from validation dataset
val_labels = [label for _, label in val_ds.unbatch()]
val_labels = [label.numpy() for label in val_labels]

# Count the occurrences of each class in the training dataset
train_counts = Counter(train_labels)
print("Training dataset class distribution:")
print(f"Abnormal: {train_counts[0]}, Normal: {train_counts[1]}")

# Count the occurrences of each class in the validation dataset
val_counts = Counter(val_labels)
print("Validation dataset class distribution:")
print(f"Abnormal: {val_counts[0]}, Normal: {val_counts[1]}")

# Plot the class distribution for the training dataset
plt.figure(figsize=(8, 3))
plt.bar(['Abnormal', 'Normal'], [train_counts[0], train_counts[1]], color='blue')
plt.title("Training Dataset Class Distribution")
plt.show()

# Plot the class distribution for the validation dataset
plt.figure(figsize=(8, 3))
plt.bar(['Abnormal', 'Normal'], [val_counts[0], val_counts[1]], color='blue')
plt.title("Validation Dataset Class Distribution")
plt.show()


# In[30]:


class_names = train_ds.class_names
for class_name in class_names:
    imgs = os.listdir(os.path.join(data_dir, class_name))[:3]
    plt.figure(figsize=(10, 10))
    for i, img in enumerate(imgs):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(plt.imread(os.path.join(data_dir, class_name, img)))
        plt.title(class_name)
        plt.axis('off')
    plt.show()


# In[7]:


# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# In[10]:


import tensorflow as tf

num_classes = 2

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(img_height, img_width, 3)),
    tf.keras.layers.Rescaling(1./255),
    
    tf.keras.layers.Conv2D(32, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    
    tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    
    tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                     min_delta=0.0001,
                                                     patience=5,
                                                     restore_best_weights=True)

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=20,
                    callbacks=[earlystop_callback])


# In[11]:


train_loss = history.history['loss']
train_acc = history.history['accuracy']
valid_loss = history.history['val_loss']
valid_acc = history.history['val_accuracy']
# Accuracy plots
plt.figure(figsize=(8, 4))
plt.plot(train_acc, color='green', linestyle='-', label='train accuracy')
plt.plot(valid_acc, color='blue', linestyle='-', label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# loss plots
plt.figure(figsize=(8, 4))
plt.plot(train_loss, color='orange', linestyle='-', label='train loss')
plt.plot(valid_loss, color='red', linestyle='-', label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[12]:


loss, acc = model.evaluate(val_ds, batch_size=batch_size)
print("validation accuracy :", round(acc, 2))
print("validation loss :", round(loss,2))


# In[14]:


y_pred = [] # store predicted labels
y_true = [] # store true labels
# iterate over the dataset
for image_batch, label_batch in val_ds:
    # use dataset.unbatch() with repea
# append true labels
    y_true.append(label_batch)
# compute predictions
    preds = model.predict(image_batch)
# append predicted labels
    y_pred.append(np.argmax(preds, axis = - 1))
# convert the true and predicted labels into tensors
correct_labels = tf.concat([item for item in y_true], axis = 0)
predicted_labels = tf.concat([item for item in y_pred], axis = 0)


# In[15]:


cm = confusion_matrix(correct_labels, predicted_labels, normalize='true')
sns.heatmap(cm, annot=True, cmap='viridis', cbar=None)
plt.title("Confusion matrix", fontweight='bold')
plt.ylabel("True", fontsize=14)
plt.xlabel("Predicted", fontsize=14)
plt.show()


# In[16]:


print(classification_report(correct_labels, predicted_labels))


# In[19]:


model.save("C:/Users/HOME/Desktop/model.keras")


# In[20]:


def prediction(img):
    class_names = ['Abnormal', 'Normal']
    my_image = load_img(img, target_size=(img_height, img_width))
    my_image = img_to_array(my_image)
    my_image = np.expand_dims(my_image, 0)
    out = np.round(model.predict(my_image)[0], 2)
    fig = plt.figure(figsize=(7, 4))
    plt.barh(class_names, out, color='lightgray', edgecolor='red', linewidth=1)
    for index, value in enumerate(out):
        plt.text(value/2 + 0.1, index, f"{100*value:.2f}%", fontweight='bold')
    plt.xticks([])
    plt.yticks([0, 1], labels=class_names, fontweight='bold', fontsize=14)
    fig.savefig('pred_img.png', bbox_inches='tight')
    return plt.show()


# In[22]:


####### Prediction on single Image
img = "C:/Users/HOME/Desktop/project2/abnormal/normal_0_682.jpg"
prediction(img)


# In[23]:


####### Prediction on single Image
img = "C:/Users/HOME/Desktop/project2/normal/normal_1_1092.jpg"
prediction(img)


# In[34]:


import tensorflow as tf
import joblib

joblib.dump(model,"C:/Users/HOME/Desktop/model.pkl")


# In[32]:


model.save("model.h5")


# In[35]:


import os
files = os.listdir(os.getcwd())
print(files)


# In[ ]:




