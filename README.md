## **CNN Image Classification using CIFAR-10 Dataset**  

### **Overview**  
This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow and Keras** to classify images from the **CIFAR-10 dataset**. The dataset consists of **60,000 images** belonging to **10 different categories** such as airplanes, automobiles, birds, cats, etc. The model is trained to recognize and classify these images with high accuracy.  

---

## **Project Structure**  
```
├── cnn_image_classification.py  # Main Python script
├── README.md                    # Project documentation
├── requirements.txt              # Required Python dependencies
└── dataset/                      # CIFAR-10 dataset (downloaded automatically)
```

---

# **CNN Image Classification using CIFAR-10 Dataset**  

## **Overview**  
This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow and Keras** to classify images from the **CIFAR-10 dataset**. The dataset consists of **60,000 images** belonging to **10 different categories** such as airplanes, automobiles, birds, cats, etc. The model is trained to recognize and classify these images with high accuracy.  

---

## **Dataset**  
The CIFAR-10 dataset contains **60,000 images** of **size 32x32 pixels** in **10 classes**:  

| Class Label  | Example Image  |
|-------------|---------------|
| **Airplane** | <img src="https://github.com/user-attachments/assets/36bc0363-cc19-4378-92fb-60a60f239867" width="100"> |
| **Automobile** | <img src="https://github.com/user-attachments/assets/23438360-1403-418b-b802-b174837c415c" width="100"> |
| **Bird** | <img src="https://github.com/user-attachments/assets/0ca22179-faf1-4c65-97ea-1b3542904360" width="100"> |
| **Cat** | <img src="https://github.com/user-attachments/assets/44c9472f-424a-4a41-856f-524a5a52de51" width="100"> |
| **Deer** | <img src="https://github.com/user-attachments/assets/24f17204-5a22-40ef-8e76-08040e0425b8" width="100"> |
| **Dog** | <img src="https://github.com/user-attachments/assets/69b3f8f6-2546-4ffc-92e7-6a7eb3c80b8b" width="100"> |
| **Frog** | <img src="https://github.com/user-attachments/assets/4a4b0cc1-f612-479a-977a-3fc84918a2e2" width="100"> |
| **Horse** | <img src="https://github.com/user-attachments/assets/266778fc-ea84-4b92-9661-a809be22f456" width="100"> |
| **Ship** | <img src="https://github.com/user-attachments/assets/65b3fff0-d1f9-4b76-a0f8-1ff8df091c1f" width="100"> |
| **Truck** | <img src="https://github.com/user-attachments/assets/15b1f145-3ede-43c4-a667-c8c5e6664090" width="100"> |

Each image is an RGB image (3 channels), and the labels are stored as numerical values from **0 to 9**, corresponding to the classes listed above.  

---

Now, your images will appear **smaller and neatly arranged** in the README. 🚀 Let me know if you need any modifications! 😊


## **Requirements**  
Before running the script, install the required dependencies:  

```bash
pip install -r requirements.txt
```

Alternatively, install manually:  
```bash
pip install tensorflow matplotlib numpy
```

---

## **How to Run**  
1. Clone this repository or copy the script.  
2. Run the script using:  
   ```bash
   python cnn_image_classification.py
   ```
3. The script will:  
   - Load and preprocess the CIFAR-10 dataset  
   - Display sample images with their class labels  
   - Train a CNN model with **three convolutional layers**  
   - Evaluate model performance on the test dataset  
   - Predict a random image's classification  

---

## **Model Architecture**  
The CNN model consists of the following layers:  

| Layer Type          | Details                      |
|---------------------|----------------------------|
| **Conv2D**         | 32 filters, (3x3) kernel, ReLU activation |
| **MaxPooling2D**   | (2x2) pool size |
| **Conv2D**         | 64 filters, (3x3) kernel, ReLU activation |
| **MaxPooling2D**   | (2x2) pool size |
| **Conv2D**         | 64 filters, (3x3) kernel, ReLU activation |
| **Flatten**        | Converts feature maps into a vector |
| **Dense**          | 64 neurons, ReLU activation |
| **Dense**          | 10 neurons (output), Softmax activation |

The **Adam optimizer** and **sparse categorical cross-entropy loss** are used for training.  

---

## **Results & Accuracy**  
After training for **10 epochs**, the model achieves a decent accuracy on the test dataset.  

Example Output:  
```
Loss: 0.45
Accuracy: 84.2%
```
*Note: Accuracy may vary depending on dataset size and hyperparameters.*  

---

## **Sample Prediction**  
After training, the script selects a random image from the test dataset and predicts its class.  

Example Output:  
```
[[0.01 0.02 0.85 0.03 0.04 0.02 0.01 0.01 0.01 0.01]]
Predicted Class: Bird
```
*(Values represent the probability distribution over the 10 classes.)*  

---

## **Improvements & Future Work**  
- **Increase dataset size**: Training on the full CIFAR-10 dataset (instead of a reduced subset) can improve accuracy.  
- **Hyperparameter tuning**: Experiment with different optimizers, learning rates, and batch sizes.  
- **Data Augmentation**: Use transformations like flipping and rotation to improve generalization.  
- **Use Transfer Learning**: Pretrained models (ResNet, MobileNet) can significantly boost performance.  

---

## **License**  
This project is open-source and free to use.  

---

This README is structured and professional, making it easy for others to understand and use your project. Let me know if you need any modifications! 🚀😊
