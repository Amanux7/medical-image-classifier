# Medical Image Classifier

This project is a web-based medical image classifier for detecting diseases from chest X-ray images. It is built using Streamlit and PyTorch.

## Features

*   **Binary Classification:** Distinguishes between "Normal" and "Pneumonia" chest X-rays.
*   **Multi-Class Classification:** Identifies "Normal", "Pneumonia", "COVID-19", and "Tuberculosis (TB)" from chest X-rays.
*   **Web Interface:** An easy-to-use interface built with Streamlit to upload images and view predictions.
*   **Deep Learning Model:** Utilizes a ResNet-18 model for classification.

## How to Use

1.  **Installation:**
    *   Clone this repository.
    *   Install the required dependencies:
        ```bash
        pip install streamlit torch torchvision Pillow
        ```

2.  **Running the Application:**
    *   Run the Streamlit app using the following command:
        ```bash
        streamlit run app.py
        ```
    *   Open your web browser and navigate to the provided URL (usually `http://localhost:8501`).

3.  **Making Predictions:**
    *   Select the classification mode: "Binary" or "Multi-Class".
    *   Upload a chest X-ray image (in `.jpg`, `.png`, or `.jpeg` format).
    *   The application will display the uploaded image and the predicted class.

## Models

This project uses two pre-trained models:

*   `binary_model.pth`: For binary classification (Normal vs. Pneumonia).
*   `multi_model.pth`: For multi-class classification (Normal, Pneumonia, COVID-19, TB).

The models are based on the ResNet-18 architecture.

## Dependencies

*   streamlit
*   torch
*   torchvision
*   Pillow (PIL)
