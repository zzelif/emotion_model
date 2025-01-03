from data_processing import load_au_data, match_images_with_au, load_images, preprocess_data, load_cached_data, save_cached_data
from model import build_hybrid_microexpression_model, build_frozen_mobilenetv2, finetune_built_mobilenetv2, build_combine_cnns
from training import train_model
from evaluation import plot_training_history, evaluate_model, predict_emotions_in_directory, plot_confusion_matrix
from datetime import datetime
# import numpy as np

# Define paths and constants
train_data_dir = "Dataset/Train"
test_data_dir = "Dataset/Test"
au_data_path = "action_units/aggregate report/normalized_final_au_image_data.csv"
directory_path = "Dataset/Train/Happy"
cache_path = "utils/preprocessed_data.pkl"
img_size = (48, 48)
batch_size = 32
epochs = 25
num_unfrozen_layers = 40
learning_rate = 1e-5
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# # Load and preprocess data
# au_data = load_au_data(au_data_path)
# matched_filenames, au_features, au_features_dict = match_images_with_au(train_data_dir, au_data)
#
# images, labels = load_images(train_data_dir, matched_filenames, img_size)
# mobnet_images, mobnet_labels = load_images(train_data_dir, matched_filenames, img_size=(224, 224))
#
# x_train_img, x_val_img, x_train_au, x_val_au, y_train, y_val, label_map = preprocess_data(images, labels, au_features)
# mob_x_train_img, mob_x_val_img, _, _, mob_y_train, mob_y_val, mob_label_map = preprocess_data(mobnet_images, mobnet_labels, None)

# Try to load cached data
cached_data = load_cached_data(cache_path)

if cached_data:
    print("Loaded cached data.")
    x_train_img, x_val_img, x_train_au, x_val_au, y_train, y_val, label_map, mob_x_train_img, mob_x_val_img, mob_y_train, mob_y_val, mob_label_map,\
        au_features, au_features_dict, matched_filenames = cached_data
    # print(f"Cached data shapes:")
    # for item in cached_data:
    #     if isinstance(item, np.ndarray):
    #         print(item.shape, item.dtype)
    #     else:
    #         print(type(item))
else:
    print("Processing data...")

    au_data = load_au_data(au_data_path)
    matched_filenames, au_features, au_features_dict = match_images_with_au(train_data_dir, au_data)

    images, labels = load_images(train_data_dir, matched_filenames, img_size)
    mobnet_images, mobnet_labels = load_images(train_data_dir, matched_filenames, img_size=(224, 224))

    x_train_img, x_val_img, x_train_au, x_val_au, y_train, y_val, label_map = preprocess_data(images, labels, au_features)
    mob_x_train_img, mob_x_val_img, _, _, mob_y_train, mob_y_val, mob_label_map = preprocess_data(mobnet_images, mobnet_labels, None)

    # Save data to cache
    save_cached_data(cache_path, (x_train_img, x_val_img, x_train_au, x_val_au, y_train, y_val, label_map, mob_x_train_img, mob_x_val_img, mob_y_train, mob_y_val, mob_label_map,
                                  au_features, au_features_dict, matched_filenames))
    print("Data cached.")

# Build and train the model
print("building the hybrid microexpression model...")
model = build_hybrid_microexpression_model(img_size + (3,), (au_features.shape[1],), len(label_map))
print("training the hybrid microexpression model...")
history = train_model(model, [x_train_img, x_train_au], y_train, [x_val_img, x_val_au], y_val, batch_size, epochs,
                      save_path="models/hybrid_micro_model.h5")

print("building the frozen mobilenetv2 model...")
mobilenet_model = build_frozen_mobilenetv2((224, 224, 3), len(label_map))
print("training the frozen mobilenetv2 model...")
mob_history = train_model(mobilenet_model, [mob_x_train_img, None], mob_y_train, [mob_x_val_img, None], mob_y_val, batch_size, epochs,
                          save_path="models/frozen_mobilenet_model.h5")
# print("Before fine-tuning:")
# for layer in mobilenet_model.layers[-5:]:
#     print(f"{layer.name} - Trainable: {layer.trainable}")

print("fine-tuning the frozen mobilenetv2 model...")
finetune_mobilenet_model = finetune_built_mobilenetv2(mobilenet_model, num_unfrozen_layers, learning_rate)
print("training the finetuned mobilenetv2 model...")
finetune_history = train_model(finetune_mobilenet_model, [mob_x_train_img, None], mob_y_train, [mob_x_val_img, None], mob_y_val, batch_size, epochs,
                               save_path="models/finetuned_mobilenetv2_model.h5")
# print("After fine-tuning:")
# for layer in finetune_mobilenet_model.layers[-5:]:
#     print(f"{layer.name} - Trainable: {layer.trainable}")

#Evaluate the model
mobilenet_features_train = finetune_mobilenet_model.predict(mob_x_train_img)
mobilenet_features_val = finetune_mobilenet_model.predict(mob_x_val_img)

combined_train_inputs = [mobilenet_features_train, x_train_img, x_train_au]
combined_val_inputs = [mobilenet_features_val, x_val_img, x_val_au]

print("building the combined cnns model...")
combined_cnns_model = build_combine_cnns(
    mobilenet_features_train.shape[1:],
    x_train_img.shape[1:],
    (au_features.shape[1],),
    len(label_map)
)
print("training the combined cnns model...")
final_history = train_model(combined_cnns_model, [combined_train_inputs], y_train, [combined_val_inputs], y_val, batch_size, epochs,
                            save_path="models/combined_finalcnn_model.h5")

print(f"Evaluating hybrid microexpression...")
accuracy, precision, recall = evaluate_model(model, [x_val_img, x_val_au], y_val)
print(f"Validation Accuracy: {accuracy:.2f}")
print(f"Validation Precision: {precision:.2f}")
print(f"Validation Recall: {recall:.2f}")

print(f"Evaluating frozen mobilenetv2...")
accuracy, precision, recall = evaluate_model(mobilenet_model, [mob_x_val_img], mob_y_val)
print(f"Validation Accuracy: {accuracy:.2f}")
print(f"Validation Precision: {precision:.2f}")
print(f"Validation Recall: {recall:.2f}")

print(f"Evaluating finetuned mobilenetv2...")
accuracy, precision, recall = evaluate_model(finetune_mobilenet_model, [mob_x_val_img], mob_y_val)
print(f"Validation Accuracy: {accuracy:.2f}")
print(f"Validation Precision: {precision:.2f}")
print(f"Validation Recall: {recall:.2f}")

print(f"Evaluating final model...")
accuracy, precision, recall = evaluate_model(combined_cnns_model, combined_val_inputs, mob_y_val)
print(f"Validation Accuracy: {accuracy:.2f}")
print(f"Validation Accuracy: {precision:.2f}")
print(f"Validation Recall: {recall:.2f}")

# Save training history
plot_training_history(
    histories=[history, mob_history, finetune_history, final_history],
    labels=['Hybrid Microexpression Model', 'Frozen MobileNetV2 Model', 'Finetuned MobileNetV2 Model', 'Combined CNNS Model'],
    output_path=f"metrics/combined_training_history_{timestamp}.png"
)

plot_confusion_matrix(combined_cnns_model, combined_val_inputs, y_val, label_map,
                      title="Combined CNNs Confusion Matrix")

#Predict from the directory path
print("Predicting using hybrid micro model...")
predict_emotions_in_directory(directory_path, model, au_features_dict, label_map, img_size=(48, 48))
print("Predicting using frozen mobilenetv2 model...")
predict_emotions_in_directory(directory_path, mobilenet_model, au_features_dict, label_map, img_size=(224, 224))
print("Predicting using finetuned mobilenetv2 model...")
predict_emotions_in_directory(directory_path, finetune_mobilenet_model, au_features_dict, label_map, img_size=(224, 224))
print("Predicting using final model...")
predict_emotions_in_directory(directory_path, combined_cnns_model, au_features_dict, label_map, img_size=(224, 224))
