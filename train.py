import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Corrected import

# Image size and batch size
img_size = (128, 128)
batch_size = 16

# RGB to grayscale images (rescale pixel values)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Read training data
train_data = train_datagen.flow_from_directory(
    'data/train',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Read validation data (corrected path)
val_data = val_datagen.flow_from_directory(
    'data/validation',  # Corrected to use validation folder
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Number of classes
num_classes = len(train_data.class_indices)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

# Save the model
model.save("cnn_classifier.h5")

# Print model summary
model.summary()
