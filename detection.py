import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import numpy as np


# Load the COCO dataset
coco_dataset, info = tfds.load('coco/2017', split='train', with_info=True)

# Define a preprocessing function
def preprocess_data(data):
    image = tf.image.resize(data['image'], (224, 224))
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values
    label = tf.cast(data['objects']['label'], tf.float32)
    return image, label

# Apply preprocessing to the dataset
train_dataset = coco_dataset.map(preprocess_data)
train_dataset = train_dataset.batch(32).shuffle(1000)

# Define the model
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=5)

# Save the trained model
model.save('rhino_detection_model.h5')

# Load the trained model
model = tf.keras.models.load_model('rhino_detection_model.h5')

# Load and preprocess the test image (rhino.jpg)
image_path = 'rhino.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))
image = image / 255.0  # Normalize pixel values

# Expand dimensions to match model's expected shape
image = np.expand_dims(image, axis=0)

# Make predictions
predictions = model.predict(image)

# Threshold the predictions (adjust the threshold based on your needs)
threshold = 0.5
prediction_label = 1 if predictions[0][0] >= threshold else 0

# Print the result
if prediction_label == 1:
    print("The model predicts: Rhino Detected!")
else:
    print("The model predicts: No Rhino Detected.")
