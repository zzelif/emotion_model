import keras.losses
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2
import os

# Set paths for datasets
train_data_dir = 'Dataset-DS/Train'
test_data_dir = 'Dataset-DS/Train'

# Define image size and batch size
img_size = (48, 48)
batch_size = 32

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Data generators
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

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Emotion labels
emotion_labels = ['Happiness', 'Anger', 'Sadness', 'Surprise', 'Neutrality']


# ------------------------------
# 0. U-Net Segmentation CNN
# ------------------------------
def build_unet_model(input_size=(48, 48, 3)):
    inputs = layers.Input(input_size)

    # Encoder: downsampling
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='SAME')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='SAME')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='SAME')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='SAME')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    # Bottleneck
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='SAME')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='SAME')(conv3)

    # Decoder: upsampling
    up1 = layers.UpSampling2D((2, 2))(conv3)
    concat1 = layers.concatenate([up1, conv2], axis=3)
    conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='SAME')(concat1)
    conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='SAME')(conv4)

    up2 = layers.UpSampling2D((2, 2))(conv4)
    concat2 = layers.concatenate([up2, conv1], axis=3)
    conv5 = layers.Conv2D(64, (3, 3), activation='relu', padding='SAME')(concat2)
    conv5 = layers.Conv2D(64, (3, 3), activation='relu', padding='SAME')(conv5)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    return model

# ------------------------------
# 1. Microexpression CNN
# ------------------------------
def build_microexpression_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='SAME', input_shape=(48, 48, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu', padding='SAME'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='SAME'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(emotion_labels), activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


# ------------------------------
# 2. MobileNetV2 Transfer Learning
# ------------------------------
def build_mobilenet_model():
    base_mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

    for layer in base_mobilenet.layers:
        layer.trainable = False

    model = Sequential([
        base_mobilenet,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(len(emotion_labels), activation='softmax')
    ])

    for layer in base_mobilenet.layers[-50:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


# ------------------------------
# 3. Combined Model
# ------------------------------
def build_combined_model():
    combined_input = Input(shape=(48, 48, 3))
    micro_model = build_microexpression_model()
    mobile_model = build_mobilenet_model()

    micro_out = micro_model(combined_input)
    mobile_out = mobile_model(combined_input)

    combined_output = tf.keras.layers.average([micro_out, mobile_out])
    combined_model = Model(inputs=combined_input, outputs=combined_output)

    combined_model.compile(optimizer=Adam(learning_rate=0.0001),
                           loss=keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])
    return combined_model


# ------------------------------
# 4. Training and Evaluation
# ------------------------------
def train_and_evaluate_model():
    # Build combined model
    model = build_combined_model()

    # Early stopping and model checkpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

    # Train the model
    model.fit(
        train_generator,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        validation_data=validation_generator,
        callbacks=[early_stopping, checkpoint]
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc}")

    return model


# Train and get the trained model
model = train_and_evaluate_model()


# ------------------------------
# 5. Realtime Emotion Detection using Webcam
# ------------------------------
def segment_face(image, unet_model):
    # Assuming the image is already preprocessed to the correct format
    img_resized = cv2.resize(image, (48, 48))  # Change size to match emotion model's input size

    img_normalized = np.expand_dims(img_resized, axis=0) / 255.0  # Normalize the image
    segmented_face = unet_model.predict(img_normalized)  # Add batch dimension
    segmented_face = np.squeeze(segmented_face, axis=0)  # Remove batch dimension
    return segmented_face


def detect_emotion_from_segmented_face(segmented_face, emotion_model):
    # Resize the segmented face to the required input size for the emotion model (48, 48, 3)
    img_resized = cv2.resize(segmented_face, (48, 48))  # Resize to 48x48

    # If the image is grayscale (1 channel), convert it to 3 channels (RGB)
    if img_resized.ndim == 2:  # Grayscale image
        img_resized = np.stack([img_resized] * 3, axis=-1)  # Convert to 3 channels

    img_normalized = np.expand_dims(img_resized, axis=0) / 255.0  # Normalize and add batch dimension (1, 48, 48, 3)

    # Predict the emotion from the segmented face
    predictions = emotion_model.predict(img_normalized)
    predicted_emotion = emotion_labels[np.argmax(predictions)]  # Get the emotion label with the highest probability
    confidence_level = predictions[0][np.argmax(predictions)]
    return predicted_emotion, confidence_level


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
    # Get all image file paths in the directory
    image_paths = [os.path.join(directory_path, fname) for fname in os.listdir(directory_path) if
                   fname.endswith(('jpg', 'png', 'jpeg'))]

    # Loop through each image and predict the emotion
    for image_path in image_paths:
        predicted_emotion, confidence_level = predict_emotion_from_image(image_path, emotion_model)

        # Print the result
        print(f"Image: {image_path}")
        print(f"Predicted Emotion: {predicted_emotion} ({confidence_level * 100:.2f}%)")
        print("-" * 50)


# Example usage:
# Assuming you have the trained emotion model (e.g., `model`)
directory_path = 'Dataset-DS/Train/Happy'  # Replace with your directory containing images
predict_emotions_in_directory(directory_path, model)

# ------------------------------
# 5. Realtime Emotion Detection using Webcam
# ------------------------------
def realtime_emotion_detection(model):
    cap = cv2.VideoCapture(0)
    unet_model = build_unet_model()
    emotion_model = build_combined_model()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for U-Net (you might need to resize or normalize the image here)
        face_segmented = segment_face(frame, unet_model)  # Get the segmented face using U-Net

        # Detect emotion from the segmented face
        predicted_emotion, confidence_level = detect_emotion_from_segmented_face(face_segmented, emotion_model)

        # Display the predicted emotion on the frame
        cv2.putText(frame, f"Emotion: {predicted_emotion} ({confidence_level*100:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Emotion Detection with Segmentation", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close the OpenCV window

# Uncomment to run the realtime detection
#realtime_emotion_detection(model)
