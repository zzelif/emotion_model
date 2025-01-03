from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, Concatenate, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNetV2

def build_hybrid_microexpression_model(input_shape_image, input_shape_au, num_classes):
    img_input = Input(shape=input_shape_image, name="image_input")
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)

    au_input = Input(shape=input_shape_au, name="au_input")
    y = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(au_input)
    y = BatchNormalization()(y)
    y = Dense(32, activation='relu')(y)

    combined = Concatenate()([x, y])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.5)(z)
    z = Dense(num_classes, activation='softmax')(z)

    model = Model(inputs=[img_input, au_input], outputs=z)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall'])
    return model

def build_frozen_mobilenetv2(input_shape_image, num_classes):
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape_image)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall'])
    return model

def finetune_built_mobilenetv2(model, num_unfrozen_layers, learning_rate):
    model.trainable = True

    for layer in model.layers[:-num_unfrozen_layers]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    return model


def build_combine_cnns(mobilenet_output_shape, input_shape_image, input_shape_au, num_classes):
    mobilenet_output = Input(shape=mobilenet_output_shape, name="mobilenet_output")
    img_input = Input(shape=input_shape_image, name="image_input")
    au_input = Input(shape=input_shape_au, name="au_input")

    flt_img_input = Flatten()(img_input)

    combined = Concatenate()([mobilenet_output, flt_img_input, au_input])
    x = Dense(128, activation='relu')(combined)
    x = Dense(64, activation='relu')(x)
    final_output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[mobilenet_output, img_input, au_input], outputs=final_output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    model.summary()

    return model
