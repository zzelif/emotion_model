import keras.losses
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, \
    GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
import os
import cv2
from collections import Counter

# Adjust data augmentation
img_size = (48, 48)
batch_size = 32
train_data_dir = 'Dataset/Train_1'

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

emotion_labels = list(train_generator.class_indices.keys())

# SMOTE Handling
images, labels = [], []
for i in range(len(train_generator)):
    batch_images, batch_labels = train_generator[i]
    images.append(batch_images)
    labels.append(batch_labels)
    if len(images) * batch_size >= train_generator.samples:
        break

images = np.vstack(images)
labels = np.argmax(np.vstack(labels), axis=1)
x_train_flat = images.reshape((images.shape[0], -1))

smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x_train_flat, labels)
x_resampled = x_resampled.reshape((-1, img_size[0], img_size[1], 3))
y_resampled = tf.keras.utils.to_categorical(y_resampled, num_classes=len(emotion_labels))


# Models
def build_microexpression_model():
    inputs = Input(shape=(48, 48, 3))
    x = Conv2D(32, (3, 3), activation='relu', padding='SAME')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='SAME')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='SAME')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(emotion_labels), activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy', 'Precision', 'Recall'])
    return model


def build_mobilenet_model():
    base_mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    for layer in base_mobilenet.layers:
        layer.trainable = False

    inputs = Input(shape=(48, 48, 3))
    resized_inputs = layers.Lambda(lambda x: tf.image.resize(x, (128, 128)))(inputs)
    features = base_mobilenet(resized_inputs)
    x = GlobalAveragePooling2D()(features)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(emotion_labels), activation='softmax')(x)

    for layer in base_mobilenet.layers[-100:]:
        layer.trainable = True

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy', 'Precision', 'Recall'])
    return model


def build_combined_model():
    input_layer = Input(shape=(48, 48, 3))
    micro_model = build_microexpression_model()
    mobile_model = build_mobilenet_model()

    micro_out = micro_model(input_layer)
    mobile_out = mobile_model(input_layer)

    combined = layers.Concatenate()([micro_out, mobile_out])
    combined_output = Dense(len(emotion_labels), activation='softmax')(combined)
    model = Model(inputs=input_layer, outputs=combined_output)

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy', 'Precision', 'Recall'])
    return model

#Shows figure of graph for metrics
def plot_metrics(history):
    plt.figure(figsize=(12,4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Training & Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')  # Save the figure, then close. Use plt.show() for immediate show
    plt.close()

#Train and Evaluate the model
def train_and_evaluate_model():
    model = build_combined_model()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr * tf.math.exp(-0.1) if epoch >= 10 else lr)

    history = model.fit(
        x_resampled, y_resampled,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        validation_data=validation_generator,
        callbacks=[early_stopping, checkpoint, lr_scheduler]
    )

    val_predictions = model.predict(validation_generator)
    val_labels = validation_generator.classes
    predicted_classes = np.argmax(val_predictions, axis=1)

    # Evaluate the model
    test_loss, test_acc, test_precision, test_recall = model.evaluate(validation_generator)
    print(f"Test Accuracy: {test_acc}")
    print(f"Test Precision: {test_precision}")
    print(f"Test Recall: {test_recall}")

    print("Classification Report:")
    print(classification_report(val_labels, predicted_classes, target_names=emotion_labels))
    print("Confusion Matrix:")
    print(confusion_matrix(val_labels, predicted_classes))

    plot_metrics(history)

    return model

trained_model = train_and_evaluate_model()

#Preprocessing for the Prediction
def load_and_preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Resize to the required input size for the emotion model (48, 48)
    img_resized = cv2.resize(img, (48, 48))

    # If the image is grayscale (1 channel), convert it to 3 channels (RGB)
    if img_resized.ndim == 2:  # Grayscale image
        img_resized = np.stack([img_resized] * 3, axis=-1)  # Convert to 3 channels

    # Normalize and add batch dimension (1, 48, 48, 3)
    img_normalized = np.expand_dims(img_resized, axis=0) / 255.0
    return img_normalized

def predict_emotion_from_image(image_path, emotion_model):
    # Load and preprocess the image
    img_normalized = load_and_preprocess_image(image_path)

    # Predict the emotion from the image
    predictions = emotion_model.predict(img_normalized)

    # Get the emotion label with the highest probability
    predicted_class = np.argmax(predictions)
    predicted_emotion = emotion_labels[predicted_class]

    # Get the confidence level (probability of the predicted class)
    confidence_level = predictions[0][predicted_class]

    return predicted_emotion, confidence_level


def predict_emotions_in_directory(directory_path, emotion_model):
    # Initialize a Counter
    emotion_tally = Counter()

    # Get all image file paths in the directory
    image_paths = [os.path.join(directory_path, fname) for fname in os.listdir(directory_path) if
                   fname.endswith(('jpg', 'png', 'jpeg'))]

    # Loop through each image and predict the emotion
    for image_path in image_paths:
        predicted_emotion, confidence_level = predict_emotion_from_image(image_path, emotion_model)

        emotion_tally[predicted_emotion] += 1

        # Print the result
        print(f"Image: {image_path}")
        print(f"Predicted Emotion: {predicted_emotion} ({confidence_level * 100:.2f}%)")
        print("-" * 50)

    print("Emotion Tally:")
    for emotion, count in emotion_tally.items():
        print(f"{emotion}: {count}")

    return emotion_tally


# Example usage:
# Assuming you have the trained emotion model (e.g., `model`)
directory_path = 'Dataset/Train_1/happy'  # Replace with your directory containing images
predict_emotions_in_directory(directory_path, trained_model)