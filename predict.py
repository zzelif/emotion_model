import os.path
from tensorflow.keras.models import load_model
from data_processing import load_au_data, match_images_with_au, load_images, preprocess_data
from model import build_hybrid_microexpression_model, build_finetuned_mobilenetv2, finetune_built_mobilenetv2
from training import train_model
from evaluation import plot_training_history, evaluate_model, predict_emotions_in_directory

# Define paths and constants
train_data_dir = "Dataset/Train"
test_data_dir = "Dataset/Test"
au_data_path = "action_units/aggregate report/normalized_final_au_image_data.csv"
directory_path = "Dataset/Test/Angry"
model_path = "models/hybrid_micro_model.h5"
img_size = (48, 48)
batch_size = 32
epochs = 25
num_unfrozen_layers = 40
learning_rate = 1e-5

# Check if model path exists and load the model
if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}. Exiting...")
    exit()

print("Loading trained Hybrid model...")
mobilenet_model = load_model(model_path)

# Load and preprocess data
au_data = load_au_data(au_data_path)
matched_filenames, au_features, au_features_dict = match_images_with_au(train_data_dir, au_data)

images, labels = load_images(train_data_dir, matched_filenames, img_size)
mobnet_images, mobnet_labels = load_images(train_data_dir, matched_filenames, img_size=(224, 224))

x_train_img, x_val_img, x_train_au, x_val_au, y_train, y_val, label_map = preprocess_data(images, labels, au_features)
mob_x_train_img, mob_x_val_img, _, _, mob_y_train, mob_y_val, mob_label_map = preprocess_data(mobnet_images, mobnet_labels, None)

# Perform predictions on the specified directory
print("Predicting emotions in directory...")
try:
    predict_emotions_in_directory(
        directory_path, mobilenet_model, au_features_dict, label_map, img_size=img_size
    )
    print("Prediction completed successfully.")
except Exception as e:
    print(f"Error during prediction: {e}")