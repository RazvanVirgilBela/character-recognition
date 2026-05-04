import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

# ---------------------------
# CONFIG
# ---------------------------
IMG_SIZE = 32
DATASET_PATH = "dataset"
TEST_PATH = "test_images"

# ---------------------------
# LOAD DATA
# ---------------------------
def load_data(dataset_path):
    images = []
    labels = []
    class_names = sorted(os.listdir(dataset_path))

    for label_index, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            images.append(img)
            labels.append(label_index)

    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = to_categorical(labels)

    print("\nDataset loaded:")
    print("Total samples:", len(images))
    print("Classes:", class_names)

    return images, labels, class_names

# ---------------------------
# BUILD MODEL
# ---------------------------
def build_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ---------------------------
# TRAIN
# ---------------------------
def train():
    X, y, class_names = load_data(DATASET_PATH)

    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_model(len(class_names))

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val, y_val)
    )

    model.save("model.h5")

    return model, class_names, history, X_val, y_val

# ---------------------------
# VISUALIZATION
# ---------------------------
def plot_training(history):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.legend(['Train', 'Val'])

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.legend(['Train', 'Val'])

    plt.show()

def show_predictions(model, X, y, class_names, num=10):
    plt.figure(figsize=(15,5))

    for i in range(num):
        idx = np.random.randint(0, len(X))
        img = X[idx]
        true_label = np.argmax(y[idx])

        pred = model.predict(img.reshape(1,IMG_SIZE,IMG_SIZE,1), verbose=0)
        pred_label = np.argmax(pred)

        plt.subplot(2,5,i+1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"T:{class_names[true_label]} | P:{class_names[pred_label]}")
        plt.axis('off')

    plt.show()

def plot_confusion_matrix(model, X, y, class_names):
    y_true = np.argmax(y, axis=1)
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# ---------------------------
# PREDICT SINGLE IMAGE
# ---------------------------
def predict_image(model, class_names, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error loading {img_path}")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    prediction = model.predict(img, verbose=0)
    index = np.argmax(prediction)

    print(f"{img_path} → Prediction: {class_names[index]}")
    
def predict_image_verbose(model, class_names, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not load {img_path}")
        return

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized / 255.0
    img_input = img_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    prediction = model.predict(img_input, verbose=0)[0]

    # 🔥 sortăm probabilitățile descrescător
    sorted_indices = np.argsort(prediction)[::-1]

    print(f"\n🖼️ Image: {img_path}")
    print("📊 Probabilities:")

    for i in sorted_indices:
        print(f"{class_names[i]}: {prediction[i]*100:.2f}%")

    best_index = np.argmax(prediction)
    print(f"\n✅ FINAL PREDICTION: {class_names[best_index]}")

    # 🔥 Vizualizare grafică
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Pred: {class_names[best_index]}")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.bar(class_names, prediction)
    plt.xticks(rotation=90)
    plt.title("Class Probabilities")

    plt.tight_layout()
    plt.show()

# ---------------------------
# TEST FOLDER (NEW IMAGES)
# ---------------------------
def test_on_folder(model, class_names, folder_path):
    print("\n--- TESTING NEW IMAGES ---")

    if not os.path.exists(folder_path):
        print("Test folder not found!")
        return

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)

        if os.path.isfile(img_path):
            predict_image_verbose(model, class_names, img_path)

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    model, class_names, history, X_val, y_val = train()

    # # Visualizations
    plot_training(history)
    show_predictions(model, X_val, y_val, class_names)
    plot_confusion_matrix(model, X_val, y_val, class_names)

    # Test on separate images
    test_on_folder(model, class_names, TEST_PATH)