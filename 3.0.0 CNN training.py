import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_excel("landmarks.xlsx")

# Extract labels and features
labels = df.iloc[:, 0].values  # First column contains labels
features = df.iloc[:, 1:].values  # Remaining columns contain 468 x and 468 y coordinates

# Encode labels (happy=0, neutral=1, sad=2)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Reshape into 2-channel format: (batch, 468, 1, 2)
features = features.reshape(-1, 468, 1, 2)

# Split dataset into train (70%), validation (15%), and test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).shuffle(1000)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

# Define CNN Model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 1), activation='relu', padding='same', input_shape=(468, 1, 2)),
        tf.keras.layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, (3, 1), activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 emotion classes
    ])
    return model

# Initialize model, loss function, and optimizer
model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=20, callbacks=[early_stopping])

# Evaluate model
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc * 100:.2f}%")

# Save model with validation accuracy in filename
val_acc_percentage = history.history['val_accuracy'][-1] * 100
model_filename = f"emotion_model_{val_acc_percentage:.2f}.h5"
model.save(model_filename)
print(f"Model saved as {model_filename}")

# Plot loss and accuracy
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.legend()
plt.title("Loss over epochs")

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.legend()
plt.title("Accuracy over epochs")

plt.show()