import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import random
from tqdm import tqdm
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

DATADIR = r'C:/Users/emili/Downloads/practica2/datasetClasif'
TRAIN_DIR = os.path.join(DATADIR, 'train')
TEST_DIR = os.path.join(DATADIR, 'test')
CATEGORIES = os.listdir(TRAIN_DIR)
print(CATEGORIES)

# Function to load data
def load_data(DATADIR, categories):
    data = []
    for category in categories:
        path = os.path.join(DATADIR, category)
        class_num = categories.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), 0)
                data.append([img_array, class_num])
            except Exception as e:
                print(e)
    return data

# Load training and testing data
train_data = load_data(TRAIN_DIR, CATEGORIES)
test_data = load_data(TEST_DIR, CATEGORIES)

# Function to preprocess the data
def preprocess_data(data):
    X = []
    y = []
    for features, label in data:
        X.append(features)
        y.append(label)
    X = np.array(X)  # No reshape needed
    X = X / 255.0  # Normalize the images
    y = np.array(y)
    y = to_categorical(y, num_classes=len(CATEGORIES))
    return X, y

# Preprocess the training and testing data
X_train, y_train = preprocess_data(train_data)
X_test, y_test = preprocess_data(test_data)

# Function to plot performance
def plot_performance(history):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.grid()
    plt.title('Train and Val Loss Evolution')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.legend()
    plt.grid()
    plt.title('Train and Val Accuracy')

    plt.tight_layout()
    plt.show()  # Ensure that the plot is displayed

# Function to create the model
def create_model(input_shape=None):
    if input_shape is None:
        input_shape = (48, 48, 1)

    model = Sequential()
    model.add(Conv2D(6, (5, 5), input_shape=input_shape, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (5, 5), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(CATEGORIES), activation='softmax'))

    return model

# Callbacks
es = EarlyStopping(
    monitor='val_accuracy', min_delta=0.0001, patience=10, verbose=2,
    mode='max', baseline=None, restore_best_weights=True
)
lr = ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.1, patience=5, verbose=2,
    mode='max', min_delta=1e-5, cooldown=0, min_lr=0
)

callbacks = [es, lr]

# Function to create SIFT features
def create_SIFT_features(data):
    Feature_data = np.zeros((len(data), 48, 48, 3))

    for i in range(len(data)):
        img = data[i]
        image8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(image8bit, None)

        img = cv2.drawKeypoints(image=image8bit, outImage=img, keypoints=kp, flags=4, color=(255, 0, 0))
        Feature_data[i] = img / 255.0

    return Feature_data

# Create SIFT features
X_train_SIFT = create_SIFT_features(X_train)
X_test_SIFT = create_SIFT_features(X_test)

# Create and compile the model
SIFT_model = create_model(input_shape=(48, 48, 3))
SIFT_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Train the model
SIFT_history = SIFT_model.fit(X_train_SIFT, y_train, batch_size=8, epochs=50, validation_data=(X_test_SIFT, y_test),
                              callbacks=callbacks)

# Plot performance
plot_performance(SIFT_history)

# Evaluate the model
SIFT_acc = SIFT_model.evaluate(X_test_SIFT, y_test, verbose=0)[1]
print("SIFT Accuracy :", SIFT_acc)

# Confusion Matrix and Classification Report
y_pred = np.argmax(SIFT_model.predict(X_test_SIFT), axis=-1)
y_true = np.argmax(y_test, axis=-1)

conf_mat = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred, target_names=CATEGORIES)

print("Confusion Matrix:\n", conf_mat)
print("\nClassification Report:\n", class_report)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
SIFT_model.save('SIFT_model.h5')