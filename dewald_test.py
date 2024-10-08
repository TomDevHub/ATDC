import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os


#CONSTANTS
TRAIN_LABELS_LOCATION = '/content/drive/MyDrive/Colab Notebooks/Patch Perfect Data/train_labels.csv'
TRAIN_IMAGES_LOCATION = '/content/drive/MyDrive/Colab Notebooks/Patch Perfect Data/train_images/'
TRAIN_ANNOTATION_LOCATION = '/content/drive/MyDrive/Colab Notebooks/Patch Perfect Data/train_annotations/'



#LOAD Y DATA (functions)
def load_labels(filename):
    df = pd.read_csv(filename)
    print(df.head())
    return df


def load_y_data(filename):
    df = load_labels(filename)
    y_train = []

    column_range = df['Pothole number'].count()
    for i in range(0,column_range-1):
      y_train.append((df.iat[i,0], df.iat[i,1]))

    return y_train


#LOAD Y DATA (FUNCTIONS)
def separate_values(line):
  float_string = line
  float_list = list(map(float, float_string.split()[1:]))
  float_tuple = tuple(float_list)
  print(float_tuple)
  return float_tuple



def load_x_data(filename, y_train):
  object_params = []
  for i in y_train:
    with open(f"{filename}p{y_train[0][0]}.txt",'r') as f:
      line = f.readline()
      object_params.append(separate_values(line))
  return object_params


#PREPARE DATA FOR ML MODEL
def format_y_data(y_data):
  y_train = []
  for i in y_data:
    y_train.append(i[1])
  return y_train

def prepare_data(x_data, y_data):
  y_train = format_y_data(y_data)

  x_train = np.array(x_data)
  y_train = np.array(y_data)
  return x_train, y_train


#MACHINE LEARNING MODEL
def create_model():
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Input(shape=(4,)))
  model.add(tf.keras.layers.Dense(128, activation='relu'))
  model.add(tf.keras.layers.Dense(128, activation='relu'))
  model.add(tf.keras.layers.Dense(1, activation='linear'))
  return model


def train_model(x_data, y_data):
  x_train, y_train = prepare_data(x_data, y_data)
  
  model = create_model()
  model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
  model.fit(x_train, y_train, epochs=10)
  model.save('predict_bags.keras')
      


# MAIN EXECUTION FUNCTION
def main():
    # Load data
    #x_train = load_x_data(TRAIN_ANNOTATIONS_LOCATION)
    y_train = load_y_data(TRAIN_LABELS_LOCATION)
    x_train = load_x_data(TRAIN_ANNOTATION_LOCATION, y_train)

    train_model(x_train, y_train)

if __name__ == "__main__":
    main()
    
