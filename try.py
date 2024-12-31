import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import tensorflow as tf

scaler = MinMaxScaler(feature_range=(0, 1))
img_size = (48, 48)
batch_size = 32
# Paths and Constants
train_data_dir = "Dataset/Train"
standardized_au_data_path = "action_units/aggregate report/normalized_final_au_image_data.csv"
selected_aus = [f"AU01_r", f"AU02_r", f"AU04_r", f"AU05_r", f"AU06_r", f"AU07_r", f"AU09_r", f"AU10_r", f"AU12_r", f"AU14_r", f"AU15_r", f"AU17_r", f"AU20_r", f"AU25_r", f"AU26_r"]

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
# test_datagen = ImageDataGenerator(rescale=1. / 255)

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

# Step 1: Load AU Data
au_data = pd.read_csv(standardized_au_data_path)

# Step 2: Match Images with AU Features
image_files = []
for root, dirs, files in os.walk(train_data_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Normalize the path to use forward slashes
            emotion = os.path.basename(root).lower().strip()
            relative_path = os.path.join(emotion, os.path.splitext(file)[0].lower().strip()).replace('\\', '/')
            image_files.append(relative_path)

# Normalize filenames in AU data
au_data['filename'] = au_data['filename'].str.lower().str.strip()

# Create a dictionary for efficient lookup
au_features_dict = {row['filename']: row.iloc[1:-1].values for _, row in au_data.iterrows()}
print("Sample keys in au_features_dict:", list(au_features_dict.keys())[:10])

image_au_features = []
matched_filenames = []
for file_path in image_files:
    base_filename = os.path.basename(file_path).lower().strip()
    if base_filename in au_features_dict:
        matched_filenames.append(file_path)
        image_au_features.append(au_features_dict[base_filename])

print(f"number of au features: {len(image_au_features)}")
print(f" number of matched filenames: {len(matched_filenames)}")

valid_matched_filenames = []
valid_image_au_features = []

possible_extensions = ['.jpeg', '.jpg', '.png']
for filename in matched_filenames:

    emotion, base_filename = filename.split('/')  # Extract "emotion/base_filename"
    img_path = None
    for ext in possible_extensions:
        potential_path = os.path.join(train_data_dir, emotion, base_filename + ext)
        if os.path.exists(potential_path):
            img_path = potential_path
            break # Normalize path

    if os.path.exists(img_path):
        valid_matched_filenames.append(filename)
        valid_image_au_features.append(au_features_dict[base_filename])
    else:
        print(f"AU data exists but image not found: {filename}")

matched_filenames = valid_matched_filenames
image_au_features = np.array(valid_image_au_features)

# Debug: Check matches
print(f"Matched {len(matched_filenames)} images with AU data.")
print("Sample matched_filenames:", matched_filenames[:10])

# Step 3: Load Image Data
def load_images(image_dir, matched_filenames, img_size=(48, 48)):
    images, labels = [], []
    for filename in matched_filenames:
        print(filename)
        try:
            # Extract emotion and base filename
            emotion, base_filename = filename.split('/')
            img_path = None
            for ext in possible_extensions:
                potential_path = os.path.join(train_data_dir, emotion, base_filename + ext)
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break  # Normalize path

            # Debug the constructed path
            if not os.path.exists(img_path):
                print(f"Image file not found: {img_path}")
            else:
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size) / 255.0
                images.append(img)
                labels.append(emotion)

        except ValueError:
            print(f"Invalid filename format: {filename}")
            continue
    return np.asarray(images), np.asarray(labels)

images, labels = load_images(train_data_dir, matched_filenames, img_size=img_size)
print(f"Loaded {len(images)} images with labels: {set(labels)}")

# Encode labels
label_map = {emotion: idx for idx, emotion in enumerate(sorted(set(labels)))}
encoded_labels = np.array([label_map[label] for label in labels])
y = to_categorical(encoded_labels)

# Step 4: Split Data
image_au_features = np.array(image_au_features)
x_train_img, x_val_img, x_train_au, x_val_au, y_train, y_val = train_test_split(
    images, image_au_features, y, test_size=0.2, random_state=42
)
print(f"Shape of image_au_features: {image_au_features.shape}")
print(f"Selected AUs: {selected_aus}")

x_train_img = x_train_img.astype('float32')
x_train_au = x_train_au.astype('float32')
y_train = y_train.astype('float32')
x_val_img = x_val_img.astype('float32')
x_val_au = x_val_au.astype('float32')
y_val = y_val.astype('float32')

x_train_au = scaler.fit_transform(x_train_au)
x_val_au = scaler.transform(x_val_au)

print(f"x_train_img dtype: {x_train_img.dtype}")
print(f"x_train_au dtype: {x_train_au.dtype}")
print(f"x_train_au: {x_train_au}, {x_train_au.shape}")
print(f"x_val_au: {x_val_au}, {x_val_au.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"y_val shape: {y_val.shape}")
print(f"Normalized x_train_au: {x_train_au[:5]}")
print(f"Normalized x_val_au: {x_val_au[:5]}")

print(f"x_train_au min: {x_train_au.min()}, max: {x_train_au.max()}")
print(f"x_val_au min: {x_val_au.min()}, max: {x_val_au.max()}")

# Step 5: Define Hybrid Model
def build_hybrid_model(input_shape_image=(48, 48, 3), input_shape_au=(len(selected_aus),)):
    # Image branch
    img_input = Input(shape=input_shape_image, name="image_input")
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)

    # AU branch
    au_input = Input(shape=input_shape_au, name="au_input")
    y = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(au_input)
    y = BatchNormalization()(y)
    y = Dense(32, activation='relu')(y)

    print(f"Expected AU labels: {len(selected_aus)}")
    print(f"Actual AU labels in dataset: {image_au_features.shape[1]}")  # Replace `au_labels` with your AU dataset

    au_output = Dense(len(selected_aus), activation='sigmoid', name="au_output")(y)

    # Concatenate branches
    combined = Concatenate()([x, y])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.5)(z)
    z = Dense(len(label_map), activation='softmax', name="emotion_output")(z)

    model = Model(inputs=[img_input, au_input], outputs=[z, au_output])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss={'emotion_output': 'categorical_crossentropy', 'au_output': 'binary_crossentropy'},
                  metrics={'emotion_output': 'accuracy', 'au_output': 'accuracy'})
    return model

model = build_hybrid_model()
model.summary()

# Step 6: Train the Model
history = model.fit(
    [x_train_img, x_train_au], [y_train, x_train_au],
    validation_data=(
        [x_val_img, x_val_au],  # Validation inputs
        [y_val, x_val_au]  # Validation outputs
    ),
    epochs=25,
    batch_size=32,
    verbose=1
)

def plot_training_history(history):
    """Plot training and validation accuracy and loss."""
    em_acc = history.history['emotion_output_accuracy']
    au_acc = history.history['emotion_output_accuracy']
    em_val_acc = history.history['val_emotion_output_accuracy']
    au_val_acc = history.history['val_au_output_accuracy']

    loss = history.history['loss']
    em_loss = history.history['emotion_output_loss']
    au_loss = history.history['au_output_loss']
    val_loss = history.history['val_loss']
    em_val_loss = history.history['val_emotion_output_loss']
    au_val_loss = history.history['val_au_output_loss']

    epochs_range = range(len(em_acc))

    plt.figure(figsize=(14, 10))

    # Accuracy plot
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, em_acc, label='Emotion Accuracy')
    plt.plot(epochs_range, em_val_acc, label='Val Emotion Accuracy')
    plt.legend(loc='upper right')
    plt.title('Emotion Accuracy')
    plt.suptitle('Training and Validation Accuracy')

    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, au_acc, label='AU Accuracy')
    plt.plot(epochs_range, au_val_acc, label='Val AU Accuracy')
    plt.legend(loc='upper right')
    plt.title('AU Accuracy')
    plt.suptitle('Training and Validation Accuracy')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig('train_au_metrics_with_loss_func_2_outputs.png')  # Save the figure, then close. Use plt.show() for immediate show
    plt.close()

# Call the plotting function
plot_training_history(history)

# Step 7: Evaluate the Model
loss, accuracy = model.evaluate([x_val_img, x_val_au], y_val)
print(f"Validation Accuracy: {accuracy:.2f}")


def load_and_preprocess_image(image_path, img_size=(48, 48)):
    """Load, resize, convert to RGB, and normalize the image."""
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    # Resize the image
    img_resized = cv2.resize(img, img_size)

    # Normalize pixel values and add a batch dimension
    img_normalized = np.expand_dims(img_resized, axis=0) / 255.0
    return img_normalized


def predict_emotion_from_image(image_path, model, au_features_dict, img_size=(48, 48)):
    """Predict emotion from an image and associated AU features."""
    # Preprocess the image
    img_normalized = load_and_preprocess_image(image_path, img_size)
    if img_normalized is None:
        return None, None

    img_normalized = img_normalized.astype('float32')

    # Extract the base filename to find corresponding AU features
    base_filename = os.path.splitext(os.path.basename(image_path))[0].lower().strip()

    # Check if the AU features for this image exist
    if base_filename not in au_features_dict:
        print(f"No AU features found for {base_filename}. Skipping prediction.")
        return None, None

    # Get the AU features and expand dimensions for batching
    au_features = np.expand_dims(au_features_dict[base_filename], axis=0).astype('float32')

    # Predict the emotion
    predictions = model.predict([img_normalized, au_features])
    predicted_class = np.argmax(predictions)
    confidence_level = predictions[0][predicted_class]

    # Map the predicted class back to the emotion label
    predicted_emotion = list(label_map.keys())[list(label_map.values()).index(predicted_class)]
    return predicted_emotion, confidence_level


def predict_emotions_in_directory(directory_path, model, au_features_dict, img_size=(48, 48)):
    """Predict emotions for all images in a directory and tally results."""
    emotion_tally = Counter()

    # Get all image file paths in the directory
    image_paths = [os.path.join(directory_path, fname) for fname in os.listdir(directory_path)
                   if fname.lower().endswith(('jpg', 'jpeg', 'png'))]

    for image_path in image_paths:
        predicted_emotion, confidence_level = predict_emotion_from_image(image_path, model, au_features_dict, img_size)
        if predicted_emotion is not None:
            emotion_tally[predicted_emotion] += 1

            # Print the result
            print(f"Image: {image_path}")
            print(f"Predicted Emotion: {predicted_emotion} ({confidence_level * 100:.2f}%)")
            print("-" * 50)

    print("\nEmotion Tally:")
    for emotion, count in emotion_tally.items():
        print(f"{emotion}: {count}")

    return emotion_tally


# Example usage:
directory_path = "Dataset/Train/Angry"  # Replace with your test directory path
predict_emotions_in_directory(directory_path, model, au_features_dict, img_size=(48, 48))
