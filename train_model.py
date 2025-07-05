# train_model.py

import tensorflow as tf
import tensorflow_datasets as tfds

# Load the dataset (flowers)
(train_ds), ds_info = tfds.load(
    'tf_flowers',
    split='train[:80%]',
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

val_ds = tfds.load(
    'tf_flowers',
    split='train[80%:]',
    shuffle_files=True,
    as_supervised=True
)

# Preprocess images
IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
    return image, label

train_ds = train_ds.map(preprocess).cache().shuffle(1000).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(preprocess).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training the model...")
history = model.fit(train_ds, validation_data=val_ds, epochs=5)

# Save the trained model
model.save('flower_model.keras')
print("Model trained and saved to 'flower_model/'")
