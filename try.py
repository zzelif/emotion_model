import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from collections import Counter
import tensorflow as tf

# Paths and Constants
train_data_dir = "Dataset/Train"
standardized_au_data_path = "action_units/aggregate report/normalized_final_au_image_data.csv"
img_size = (48, 48)
selected_aus = [f"AU04_r", f"AU09_r", f"AU15_r", f"AU17_r", f"AU06_r", f"AU07_r", f"AU10_r", f"AU12_r", f"AU14_r", f"AU20_r"]

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

# Step 5: Define Hybrid Model
def build_hybrid_model(input_shape_image=(48, 48, 3), input_shape_au=(len(selected_aus),)):
    # Image branch
    img_input = Input(shape=input_shape_image, name="image_input")
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    # AU branch
    au_input = Input(shape=input_shape_au, name="au_input")
    y = Dense(64, activation='relu')(au_input)
    y = BatchNormalization()(y)
    y = Dense(32, activation='relu')(y)

    # Concatenate branches
    combined = Concatenate()([x, y])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.5)(z)
    z = Dense(len(label_map), activation='softmax')(z)

    model = Model(inputs=[img_input, au_input], outputs=z)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_hybrid_model()
model.summary()

# Step 6: Train the Model
history = model.fit(
    [x_train_img, x_train_au], y_train,
    validation_data=([x_val_img, x_val_au], y_val),
    epochs=25,
    batch_size=32,
    verbose=1
)

# Step 7: Evaluate the Model
loss, accuracy = model.evaluate([x_val_img, x_val_au], y_val)
print(f"Validation Accuracy: {accuracy:.2f}")

