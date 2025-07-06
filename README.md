# Image Classification with TensorFlow

This project demonstrates an end-to-end **image classification system** using a trained **TensorFlow model**. It detects flower types like daisy, dandelion, tulips, sunflowers, and roses from uploaded images.

---

## About the Project

This AI project was built to classify flower images using a Convolutional Neural Network (CNN). It involves:

- Training a custom image classification model using **TensorFlow**
- Saving the trained model in `.keras` format
- Testing the model on real flower images
- Future goal: Integrating with a **React frontend** to make a web-based app

---

## Tech Stack

| Component     | Technology        |
|---------------|------------------|
| AI Model      | TensorFlow / Keras |
| Dataset       | `tf_flowers` (from TensorFlow Datasets) |
| Language      | Python |
| Image Format  | `.jpg` |
| Model Output  | `.keras` (can be converted to TensorFlow.js format) |

---

## Classes the Model Can Recognize

- 🌼 Daisy  
- 🌻 Dandelion  
- 🌷 Tulips  
- 🌞 Sunflowers  
- 🌹 Roses  

---

## How It Works

1. **train_model.py**  
   - Loads `tf_flowers` dataset  
   - Preprocesses and trains a CNN  
   - Saves the model as `flower_model.keras`

2. **test_custom_image.py**  
   - Loads the saved model  
   - Takes one image  
   - Predicts and prints the class and confidence

3. **flower_model.keras**  
   - This is your trained model, ready for deployment or conversion to TensorFlow.js

4. **Sample images**  
   - Images used to test or demo predictions

---

## How to Run

> Make sure you are in a Python environment (`conda` or `venv`) with TensorFlow installed.

### Step 1: Install dependencies

```bash
pip install tensorflow tensorflow-datasets
````

### Step 2: Train the model

```bash
python train_model.py
```

### Step 3: Test the model on a sample image

```bash
python test_custom_image.py
```

---

## Live App (Coming Soon)

This model will be integrated into a React-based web app using **TensorFlow\.js**, so users can classify their own flower images right from the browser.

---

## Folder Structure

```
AI project/
│
├── daisy.jpg
├── dandelion.jpg
├── roses.jpg
├── sunflowers.jpg
├── tulips.jpg
├── flower_model.keras
├── train_model.py
├── test_custom_image.py
└── README.md
```

---

## Future Improvements

* Convert `.keras` to `TensorFlow.js` and integrate into React
* Add upload feature in frontend
* Deploy on web (Netlify/Vercel + Node backend if needed)

---
