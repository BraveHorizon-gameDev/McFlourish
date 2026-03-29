import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, BatchNormalization
from tensorflow.keras.models import Model

# -----------------------
# CONFIG
# -----------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

train_dir = "dataset_classification/train"
val_dir = "dataset_classification/val"
test_dir = "dataset_classification/test"

# -----------------------
# DATA GENERATORS
# -----------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    validation_split=0.2,
    subset='training'
)

val_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    validation_split=0.2,
    subset='validation'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("Classes:", train_data.class_indices)

# -----------------------
# MODEL
# -----------------------
base_model = EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze most layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

outputs = Dense(3, activation='softmax')(x)

model = Model(inputs, outputs)

# -----------------------
# COMPILE
# -----------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------
# TRAIN
# -----------------------
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data
)

# -----------------------
# EVALUATE
# -----------------------
loss, accuracy = model.evaluate(test_data)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# -----------------------
# SAVE
# -----------------------
model.save("fire_detector_model.h5")