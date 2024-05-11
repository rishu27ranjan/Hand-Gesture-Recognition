import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # Replace with your preferred ML model

# Function to pre-process the image for gesture recognition
def pre_process_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization (optional)
    # gray_image = cv2.equalizeHist(gray_image)

    # Resize to a fixed size (consider your model's input size)
    resized_image = cv2.resize(gray_image, (100, 100))

    # Flatten the image into a feature vector
    features = resized_image.flatten()

    # Standardize the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features.reshape(1, -1))

    return features

# Function to load and pre-process your training dataset
def load_training_data(image_paths, labels):
    features = []
    for image_path, label in zip(image_paths, labels):
        # Load image
        image = cv2.imread(image_path)

        # Pre-process image
        features.append(pre_process_image(image))

    features = np.array(features)
    return features, labels

# Load your training data (replace with your image paths and labels)
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]  # Images for each gesture
labels = ["gesture1", "gesture2", ...]  # Corresponding labels for each image

# Load training data and pre-process images
X_train, y_train = load_training_data(image_paths, labels)

# Choose and train your machine learning model (replace with your preferred model)
clf = SVC(kernel='linear')  # Example: Support Vector Machine (SVM)
clf.fit(X_train, y_train)

# Function to recognize a gesture in a live video stream
def recognize_gesture(frame):
    features = pre_process_image(frame)
    prediction = clf.predict(features.reshape(1, -1))
    return prediction[0]

# Main loop for real-time gesture recognition
cap = cv2.VideoCapture(0)  # Use 1 for external camera

while True:
    ret, frame = cap.read()

    # Recognize gesture
    gesture = recognize_gesture(frame)

    # Display frame with gesture label (optional)
    cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
