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
IMAGE_FOLDER = 'train_images'
CSV_FILE = 'potholes.csv'
FILELIST = 'filelist.txt'

# Load image filenames from the filelist.txt
def load_image_filenames(filelist):
    with open(filelist, 'r') as file:
        filenames = {line.strip() for line in file.readlines()}
    return filenames

# Load labels from CSV
def load_labels(filename):
    df = pd.read_csv(filename)
    df_new = df[df['object-class'] == 0].drop(['pothole-number','x','y'],axis=1)
    print("Loaded CSV with columns:", df_new.columns)
    df_new.columns = [col.strip() for col in df_new.columns]  # Clean column names
    print("Cleaned column names:", df_new.columns)
    print(df_new.head())  # Print the first few rows to check the data
    return df_new

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
    model = Sequential()

    # Input layer (2 input features)
    model.add(Dense(10, input_dim=2, activation='relu'))

    # Hidden layer(s)
    model.add(Dense(5, activation='relu'))

    # Output layer (1 output)
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Load all data
def load_data():
    heights, widths, bags = [], [] , []
    train_labels = load_labels(CSV_FILE)
    
    for index, row in train_labels.iterrows():
        heights.append(row['height'])
        widths.append(row['width'])
        
        bags.append(row['bags-used'])
    return np.array(heights), np.array(widths),np.array(bags)

# Main execution function
def main():
    # Load and split data
    heights, widths, bags = load_data()
    width_height = np.column_stack((widths,heights))
    print(width_height)    
    print(bags)
   
    if len(width_height) > 0  and len(bags):
        
        X_train, X_test, y_train, y_test = train_test_split(width_height, bags, test_size=0.2, random_state=42)
        
        # Create and train model
        model = create_model() 
        print('Aweeeeeeeeeeeeeeeeeeeeezzzzzzzzzzzzz')
    
        model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

        # Evaluate model
        val_loss, val_mae = model.evaluate(X_test, y_test)
        print(f'Validation MAE: {val_mae}')
    else:
        print("No data available for training.")

if __name__ == "__main__":
    main()
