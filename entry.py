from data_processing import load_au_data, match_images_with_au, load_images, preprocess_data
from model import build_hybrid_microexpression_model
from training import train_model
from evaluation import plot_training_history, evaluate_model, predict_emotions_in_directory

# Define paths and constants
train_data_dir = "Dataset/Train"
test_data_dir = "Dataset/Test"
au_data_path = "action_units/aggregate report/normalized_final_au_image_data.csv"
directory_path = "Dataset/Train/Happy"
img_size = (48, 48)
batch_size = 32

# Load and preprocess data
au_data = load_au_data(au_data_path)
matched_filenames, au_features, au_features_dict = match_images_with_au(train_data_dir, au_data)
images, labels = load_images(train_data_dir, matched_filenames, img_size)
x_train_img, x_val_img, x_train_au, x_val_au, y_train, y_val, label_map = preprocess_data(images, labels, au_features)

# Build and train the model
model = build_hybrid_microexpression_model(img_size + (3,), (au_features.shape[1],), len(label_map))
history = train_model(model, x_train_img, x_train_au, y_train, x_val_img, x_val_au, y_val, batch_size)

# Evaluate the model
accuracy, precision, recall = evaluate_model(model, x_val_img, x_val_au, y_val)
print(f"Validation Accuracy: {accuracy:.2f}")
print(f"Validation Accuracy: {precision:.2f}")
print(f"Validation Accuracy: {recall:.2f}")

# Save training history
plot_training_history(history, "metrics/train_history.png")

#Predict from the directory path
predict_emotions_in_directory(directory_path, model, au_features_dict, label_map, img_size=(48, 48))
