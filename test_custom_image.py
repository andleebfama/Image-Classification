import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Class labels (same order as training)
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Load your trained model
model = tf.keras.models.load_model('flower_model.keras')

# Load your image (replace filename if needed)
img_path = 'dandelion.jpg'  # ← apni image ka naam yahan likhein
img = Image.open(img_path).convert('RGB')
img = img.resize((224, 224))

# Convert image to array
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
img_array = img_array / 255.0  # Normalize

# Predict
prediction = model.predict(img_array)
predicted_index = np.argmax(prediction[0])
confidence = np.max(prediction[0]) * 100
predicted_class = class_names[predicted_index]

# Show result
print(f"Predicted: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")

# Optional: show image with title
plt.imshow(img)
plt.title(f"{predicted_class} ({confidence:.2f}%)")
plt.axis('off')
plt.show()
