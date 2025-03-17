# Import necessary libraries from TensorFlow and other modules
from tensorflow.keras import datasets, layers
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Load the CIFAR-10 dataset (a dataset of 60,000 32x32 color images in 10 classes)
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize the pixel values (scale them to the range [0,1]) for better training performance
x_train, x_test = x_train / 255, x_test / 255

# Define the classification labels corresponding to the dataset's 10 classes
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']

# Display a few sample images from the training dataset
for i in range(4):
    plt.subplot(2, 2, i + 1)  # Arrange plots in a 2x2 grid
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    plt.imshow(x_train[i])  # Show the image
    plt.xlabel(classification[y_train[i][0]])  # Set label as the class name

plt.show()  # Display the images

# Reduce the dataset size for faster training (optional step)
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:1000]
y_test = y_test[:1000]

# Create a Convolutional Neural Network (CNN) model
model = keras.Sequential([
    # First Convolutional Layer: 32 filters of size (3x3), ReLU activation, input shape (32x32x3)
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),  # Max pooling layer to reduce dimensions

    # Second Convolutional Layer: 64 filters of size (3x3), ReLU activation
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),  # Max pooling layer

    # Third Convolutional Layer: 64 filters of size (3x3), ReLU activation
    keras.layers.Conv2D(64, (3, 3), activation='relu'),

    keras.layers.Flatten(),  # Flatten the 2D feature maps into a 1D feature vector
    keras.layers.Dense(64, activation='relu'),  # Fully connected layer with 64 neurons
    keras.layers.Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each class)
])

# Compile the model
model.compile(
    optimizer='Adam',  # Adam optimizer for adaptive learning rate
    loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']  # Evaluate performance using accuracy
)

# Train the model using the training dataset (15 epochs, with validation on test data)
model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))

# Evaluate the model's performance on the test dataset
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

# Display a test image for prediction
plt.imshow(x_test[1])

# Predict the class of the displayed image
predictions = model.predict(x_test[1].reshape(1, 32, 32, 3))  # Reshape input for model compatibility
predicted_class = np.argmax(predictions)  # Get the class with the highest probability

# Print the prediction probabilities and the predicted class label
print(predictions)  
print(classification[predicted_class])  # Print the predicted class name
