# src/preprocess.py
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2

IMG_SIZE = 128

def load_image_dataset(data_root="data/raw"):
    X = []
    y = []
    classes = sorted(os.listdir(data_root))
    class_map = {c:i for i,c in enumerate(classes)}
    for c in classes:
        d = os.path.join(data_root, c)
        for fname in os.listdir(d):
            if fname.endswith(".png"):
                img = cv2.imread(os.path.join(d, fname), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype("float32") / 255.0
                X.append(img[..., None])
                y.append(class_map[c])
    X = np.array(X)
    y = to_categorical(y, num_classes=len(classes))
    return X, y, classes

if __name__ == "__main__":
    X, y, classes = load_image_dataset()
    print("Loaded", X.shape, len(classes), "classes")
    os.makedirs("data/processed_with_v", exist_ok=True)
    np.save("data/processed_with_v/X.npy", X)
    np.save("data/processed_with_v/y.npy", y)
    with open("data/processed_with_v/classes.txt", "w") as f:
        for c in classes:
            f.write(c + "\n")

    print("Saved preprocessed data to data/processed_with_v/")