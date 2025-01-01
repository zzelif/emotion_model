from data_processing import load_au_data, match_images_with_au, load_images, preprocess_data
from model import build_hybrid_microexpression_model, build_finetuned_mobilenetv2, finetune_built_mobilenetv2
from training import train_model
from evaluation import plot_training_history, evaluate_model, predict_emotions_in_directory

# Define paths and constants
train_data_dir = "Dataset/Train"
test_data_dir = "Dataset/Test"
au_data_path = "action_units/aggregate report/normalized_final_au_image_data.csv"
directory_path = "Dataset/Train/Happy"
img_size = (48, 48)
batch_size = 32
epochs = 25
num_unfrozen_layers = 40
learning_rate = 1e-5

# Load and preprocess data
au_data = load_au_data(au_data_path)
matched_filenames, au_features, au_features_dict = match_images_with_au(train_data_dir, au_data)

images, labels = load_images(train_data_dir, matched_filenames, img_size)
mobnet_images, mobnet_labels = load_images(train_data_dir, matched_filenames, img_size=(224, 224))

x_train_img, x_val_img, x_train_au, x_val_au, y_train, y_val, label_map = preprocess_data(images, labels, au_features)
mob_x_train_img, mob_x_val_img, _, _, mob_y_train, mob_y_val, mob_label_map = preprocess_data(mobnet_images, mobnet_labels, None)

# Build and train the model
print("building the hybrid microexpression model...")
model = build_hybrid_microexpression_model(img_size + (3,), (au_features.shape[1],), len(label_map))
print("building the finetuned mobilenetv2 model...")
mobilenet_model = build_finetuned_mobilenetv2((224, 224, 3), len(label_map))

print("training the hybrid microexpression model...")
history = train_model(model, x_train_img, x_train_au, y_train, x_val_img, x_val_au, y_val, batch_size, epochs,
                      save_path="models/hybrid_micro_model.h5")
print("training the frozen mobilenetv2 model...")
mob_history = train_model(mobilenet_model, mob_x_train_img, None, mob_y_train, mob_x_val_img, None, mob_y_val, batch_size, epochs,
                          save_path="models/frozen_mobilenet_model.h5")

print("Before fine-tuning:")
for layer in mobilenet_model.layers[-5:]:
    print(f"{layer.name} - Trainable: {layer.trainable}")

print("fine-tuning the frozen mobilenetv2 model...")
finetune_mobilenet_model = finetune_built_mobilenetv2(mobilenet_model, num_unfrozen_layers, learning_rate)

print("training the finetuned mobilenetv2 model...")
finetune_history = train_model(finetune_mobilenet_model, mob_x_train_img, None, mob_y_train, mob_x_val_img, None, mob_y_val, batch_size, epochs,
                               save_path="models/finetuned_mobilenetv2_model.h5")

print("After fine-tuning:")
for layer in finetune_mobilenet_model.layers[-5:]:
    print(f"{layer.name} - Trainable: {layer.trainable}")

#Evaluate the model
print(f"Evaluating hybrid microexpression...")
accuracy, precision, recall = evaluate_model(model, x_val_img, x_val_au, y_val)
print(f"Validation Accuracy: {accuracy:.2f}")
print(f"Validation Accuracy: {precision:.2f}")
print(f"Validation Accuracy: {recall:.2f}")

print(f"Evaluating frozen mobilenetv2...")
accuracy, precision, recall = evaluate_model(mobilenet_model, mob_x_val_img, None, mob_y_val)
print(f"Validation Accuracy: {accuracy:.2f}")
print(f"Validation Accuracy: {precision:.2f}")
print(f"Validation Accuracy: {recall:.2f}")

print(f"Evaluating finetuned mobilenetv2...")
accuracy, precision, recall = evaluate_model(finetune_mobilenet_model, mob_x_val_img, None, mob_y_val)
print(f"Validation Accuracy: {accuracy:.2f}")
print(f"Validation Accuracy: {precision:.2f}")
print(f"Validation Accuracy: {recall:.2f}")

# Save training history
plot_training_history(
    histories=[history, mob_history, finetune_history],
    labels=['Hybrid Microexpression Model', 'Frozen MobileNetV2 Model', 'Finetuned MobileNetV2 Model'],
    output_path="metrics/combined_training_history.png"
)

#Predict from the directory path
print("Predicting using hybrid micro model...")
predict_emotions_in_directory(directory_path, model, au_features_dict, label_map, img_size=(48, 48))
print("Predicting using frozen mobilenetv2 model...")
predict_emotions_in_directory(directory_path, mobilenet_model, au_features_dict, label_map, img_size=(224, 224))
print("Predicting using finetuned mobilenetv2 model...")
predict_emotions_in_directory(directory_path, finetune_mobilenet_model, au_features_dict, label_map, img_size=(224, 224))
