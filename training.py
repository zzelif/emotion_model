from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def create_generators(train_data_dir, test_data_dir, img_size, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='training'
    )
    val_generator = train_datagen.flow_from_directory(
        train_data_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='validation'
    )
    test_generator = test_datagen.flow_from_directory(
        test_data_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='training'
    )
    return train_generator, val_generator, test_generator

def train_model(model, train_inputs, y_train, val_inputs, y_val, batch_size, epochs, save_path):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, mode='auto', min_lr=1e-6, verbose=1)

    if isinstance(model.input, list):
        x_inputs = train_inputs  # Use the provided inputs directly for multi-input models
        x_val = val_inputs
    else:
        x_inputs = train_inputs[0]
        x_val = val_inputs[0]

    return model.fit(
        x_inputs, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint, lr_scheduler]
    )