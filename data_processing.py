import os
import cv2
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_au_data(file_path):
    return pd.read_csv(file_path)

def match_images_with_au(train_data_dir, au_data):
    possible_extensions = ['.jpeg', '.jpg', '.png']

    image_files = []
    for root, dirs, files in os.walk(train_data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                emotion = os.path.basename(root).lower().strip()
                relative_path = os.path.join(emotion, os.path.splitext(file)[0].lower().strip()).replace('\\', '/')
                image_files.append(relative_path)

    au_data['filename'] = au_data['filename'].str.lower().str.strip()
    au_features_dict = {row['filename']: row.iloc[1:-1].values for _, row in au_data.iterrows()}
    matched_filenames, image_au_features = [], []
    for file_path in image_files:
        base_filename = os.path.basename(file_path).lower().strip()
        if base_filename in au_features_dict:
            matched_filenames.append(file_path)
            image_au_features.append(au_features_dict[base_filename])

    valid_matched_filenames, valid_image_au_features = [], []

    for filename in matched_filenames:
        emotion, base_filename = filename.split('/')
        img_path = None
        for ext in possible_extensions:
            potential_path = os.path.join(train_data_dir, emotion, base_filename + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        if os.path.exists(img_path):
            valid_matched_filenames.append(filename)
            valid_image_au_features.append(au_features_dict[base_filename])
        else:
            print(f"AU data exists but image not found: {filename}")

    matched_filenames = valid_matched_filenames
    image_au_features = np.array(valid_image_au_features)
    return matched_filenames, image_au_features, au_features_dict

def load_images(train_data_dir, matched_filenames, img_size=(48, 48)):
    possible_extensions = ['.jpeg', '.jpg', '.png']
    images, labels = [], []
    for filename in matched_filenames:
        try:
            emotion, base_filename = filename.split('/')
            img_path = None
            for ext in possible_extensions:
                potential_path = os.path.join(train_data_dir, emotion, base_filename + ext)
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break
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

def preprocess_data(images, labels, image_au_features, test_size=0.2):
    label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    encoded_labels = np.array([label_map[label] for label in labels])
    encoded_labels = to_categorical(encoded_labels)
    if image_au_features is None:
        x_train_img, x_val_img, y_train, y_val = train_test_split(
            images, encoded_labels, test_size=test_size, random_state=42
        )
        x_train_img = x_train_img.astype('float32')
        y_train = y_train.astype('float32')
        x_val_img = x_val_img.astype('float32')
        y_val = y_val.astype('float32')

        return x_train_img, x_val_img, None, None, y_train, y_val, label_map

    image_au_features = np.array(image_au_features)
    x_train_img, x_val_img, x_train_au, x_val_au, y_train, y_val = train_test_split(
        images, image_au_features, encoded_labels, test_size=test_size, random_state=42
    )
    x_train_img = x_train_img.astype('float32')
    x_train_au = x_train_au.astype('float32')
    y_train = y_train.astype('float32')
    x_val_img = x_val_img.astype('float32')
    x_val_au = x_val_au.astype('float32')
    y_val = y_val.astype('float32')

    return x_train_img, x_val_img, x_train_au, x_val_au, y_train, y_val, label_map

# Cache and Load functions
def save_cached_data(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_cached_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None