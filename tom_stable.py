import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import img_to_array

# Constants
IMAGE_SIZE = 256

# Load labels from CSV
def load_labels(filename):
    df = pd.read_csv(filename)
    print("Loaded CSV with columns:", df.columns)
    print(df.head())  # Print the first few rows to check the data
    return df


# Load image and process
def load_and_process_image(image_path):
    try:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        return img_to_array(image) / 255.0
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Create a TensorFlow model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Load all data
def load_data():
    images, labels = [], []
    train_labels = load_labels('train_labels.csv')
    train_labels.columns = [col.strip() for col in train_labels.columns]  # Clean column names
    print("Cleaned column names:", train_labels.columns)  # Debugging print

    for index, row in train_labels.iterrows():
        image_file = f"{row.get('Pothole number', 'default_id')}.jpg"  # Assuming 'Pothole number' is the column name
        image_path = os.path.join('./train_images', image_file)  # Correct path
        if os.path.exists(image_path):
            image = load_and_process_image(image_path)
            if image is not None:
                images.append(image)
                labels.append(row.get('Bags used', 0))
    return np.array(images), np.array(labels)


# Main execution function
def main():
    # Load and split data
    images, labels = load_data()
    if len(images) > 0 and len(labels) > 0:
        X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

        # Create and train model
        model = create_model()
        model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

        # Evaluate model
        val_loss, val_mae = model.evaluate(X_val, y_val)
        print(f'Validation MAE: {val_mae}')
    else:
        print("No data available for training.")

if __name__ == "__main__":
    main()
