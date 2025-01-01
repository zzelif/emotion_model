import matplotlib.pyplot as plt
from collections import Counter
import os
import numpy as np
import cv2

def plot_training_history(histories, labels, output_path):
    plt.figure(figsize=(12, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    for history, label in zip(histories, labels):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        epochs_range = range(len(acc))
        plt.plot(epochs_range, acc, label=f'{label} Training Accuracy')
        plt.plot(epochs_range, val_acc, label=f'{label} Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    for history, label in zip(histories, labels):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(len(loss))
        plt.plot(epochs_range, loss, label=f'{label} Training Loss')
        plt.plot(epochs_range, val_loss, label=f'{label} Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate_model(model, x_val_img, x_val_au, y_val):
    # Check if the model expects multiple inputs
    if isinstance(model.input, list) and len(model.input) > 1:
        inputs = [x_val_img, x_val_au]
    else:
        inputs = x_val_img  # Single input model (e.g., MobileNetV2)

    # Evaluate the model
    loss, accuracy, precision, recall = model.evaluate(inputs, y_val)
    return accuracy, precision, recall

def predict_image(image_path, model, au_features_dict, label_map, img_size):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None, None

    # Preprocess the image
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=0) / 255.0
    img = img.astype('float32')

    # Check if the model requires AU features
    if isinstance(model.input, list) and len(model.input) > 1:
        # Hybrid model requiring AU features
        base_filename = os.path.splitext(os.path.basename(image_path))[0].lower().strip()
        if base_filename not in au_features_dict:
            print(f"No AU features found for {base_filename}. Skipping prediction.")
            return None, None
        au_features = np.expand_dims(au_features_dict[base_filename], axis=0).astype('float32')
        inputs = [img, au_features]
    else:
        # Single-input model like MobileNetV2
        inputs = img

    # Make predictions
    predictions = model.predict(inputs)
    class_idx = np.argmax(predictions)

    confidence_level = predictions[0][class_idx]
    predicted_emotion = list(label_map.keys())[list(label_map.values()).index(class_idx)]
    return predicted_emotion, confidence_level

def predict_emotions_in_directory(directory_path, model, au_features_dict, label_map, img_size):
    emotion_tally = Counter()

    image_paths = [os.path.join(directory_path, fname) for fname in os.listdir(directory_path)
                   if fname.lower().endswith(('jpg', 'jpeg', 'png'))]

    print(f"Files in directory: {len(os.listdir(directory_path))}")

    for image_path in image_paths:
        print(f"Processing Image: {image_path}")
        try:
            predicted_emotion, confidence_level = predict_image(image_path, model, au_features_dict, label_map,
                                                                img_size)
            if predicted_emotion is not None:
                emotion_tally[predicted_emotion] += 1
                # Print the result
                print(f"Image: {image_path}")
                print(f"Predicted Emotion: {predicted_emotion} with confidence ({confidence_level * 100:.2f}%)")
                print("-" * 50)
            else:
                print(f"Skipping {image_path}: No matching AU features or image could not be loaded.")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    print("\nEmotion Tally:")
    for emotion, count in emotion_tally.items():
        print(f"{emotion}: {count}")

    return emotion_tally