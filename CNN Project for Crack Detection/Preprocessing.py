import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import cv2
from pathlib import Path

# Ignore warnings
warnings.filterwarnings('ignore')

# SETTING UP PATH FOR TRAIN DATA
base_path = r"C:\Users\ugonn\Downloads\Dataset"  # Define paths and labels
img_size = 224
labels = ['Non-cracks', 'Cracks']

def read_images(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
                if img_arr is not None:
                    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
                    resized_arr = cv2.resize(img_arr, (img_size, img_size))
                    data.append([resized_arr, class_num])
                else:
                    print(f"Failed to read {img} in {label} class")
            except FileNotFoundError as e:
                print(f"Error reading image: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
    return np.array(data, dtype=object)

# Load the dataset
dataset = read_images(base_path)

# DATA VISUALIZATION

# Graphing the Images --> Crack vs Non-cracks
Im = []
for label in dataset:
    if label[1] == 0:
        Im.append("Non-cracks")
    else:
        Im.append("Cracks")

plt.figure(figsize=(10, 10))
sns.set_style('darkgrid')
axl = sns.countplot(Im)
axl.set_title("Number of Images")
plt.show()

# NORMALIZATION OF DATA
x = []
y = []

for feature, label in dataset:
    x.append(feature)
    y.append(label)

x = np.array(x).reshape(-1, img_size, img_size, 3)  # RGB images have 3 channels
x = x / 255.0
y = np.array(y)

plt.subplot(1, 2, 1)
plt.imshow(x[1000])
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x[1000])
plt.axis('off')
plt.show()

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Load the pre-trained InceptionV3 model (without the final classification layers)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# Freeze the pre-trained model's weights (comment out if fine-tuning is desired)
for layer in base_model.layers:
    layer.trainable = False

# Adding custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global Average Pooling instead of Flatten for efficiency
x = Dense(1024, activation='relu')(x)  # Dense layer with 1024 units and ReLU activation
x = Dropout(0.5)(x)  # Dropout layer with 50% probability to prevent overfitting
predictions = Dense(1, activation='sigmoid')(x)  # Output layer with 1 unit and sigmoid activation for binary classification

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Define a learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Compile the model with an initial learning rate
model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

# Use the learning rate scheduler callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Train the model with the learning rate scheduler
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val), callbacks=[lr_scheduler])

# Print a summary of the model architecture
model.summary()

# Evaluate the model
y_pred = model.predict(x_val)
y_pred = np.round(y_pred).astype(int)

print("Classification Report:\n", classification_report(y_val, y_pred, target_names=labels))
conf_matrix = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Non-cracks', 'Predicted Cracks'], yticklabels=['Actual Non-cracks', 'Actual Cracks'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot training & validation accuracy and loss values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')

plt.show()

