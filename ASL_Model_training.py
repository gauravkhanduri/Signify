import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Constants
DATA_DIR = 'data/asl_dataset'
MODEL_DIR = 'models'
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 50

# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_data(data_dir):
    """
    Load ASL landmark data from CSV files
    Each row should contain the x,y,z coordinates of hand landmarks 
    and the corresponding sign label
    """
    print("Loading data...")
    
    # Placeholder for actual data loading
    # In a real scenario, you would load from files like:
    # data = pd.read_csv(os.path.join(data_dir, 'asl_landmarks.csv'))
    
    # For this example, we'll generate synthetic data
    n_samples = 1000
    n_landmarks = 21  # MediaPipe provides 21 hand landmarks
    n_features = n_landmarks * 3  # x, y, z for each landmark
    
    # Generate random landmark data
    X = np.random.rand(n_samples, n_features)
    
    # Generate random labels (A-Z, SPACE, DELETE, NOTHING)
    labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['SPACE', 'DELETE', 'NOTHING']
    y = np.random.choice(labels, n_samples)
    
    print(f"Loaded {n_samples} samples with {n_features} features")
    return X, y

def preprocess_data(X, y):
    """Preprocess the data for model training"""
    print("Preprocessing data...")
    
    # Encode the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Convert to one-hot encoding
    y_categorical = to_categorical(y_encoded)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=RANDOM_SEED
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, label_encoder

def build_model(input_shape, num_classes):
    """Build a neural network for ASL classification"""
    print("Building model...")
    
    model = Sequential([
        # Input layer
        Dense(128, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Hidden layers
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Train the model with early stopping and checkpoints"""
    print("Training model...")
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_DIR, 'asl_model_checkpoint.h5'),
        monitor='val_accuracy',
        save_best_only=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, checkpoint]
    )
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance"""
    print("Evaluating model...")
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return loss, accuracy

def save_model(model, label_encoder):
    """Save the model and label encoder"""
    print("Saving model...")
    
    # Save the model
    model_path = os.path.join(MODEL_DIR, 'asl_model.h5')
    model.save(model_path)
    
    # Save the label mapping
    labels = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open(os.path.join(MODEL_DIR, 'labels.json'), 'w') as f:
        import json
        json.dump(labels, f)
    
    print(f"Model saved to {model_path}")

def plot_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    plt.close()
    
    print(f"Training history plot saved to {os.path.join(MODEL_DIR, 'training_history.png')}")

def main():
    # Load data
    X, y = load_data(DATA_DIR)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(X, y)
    
    # Build model
    input_shape = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    model = build_model(input_shape, num_classes)
    
    # Train model
    model, history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, label_encoder)
    
    # Plot history
    plot_history(history)
    
    print("Training complete!")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    main()