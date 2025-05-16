import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import fetch_mnist, DataLoader
import matplotlib.pyplot as plt

def main():
    print("Loading...")
    X_train, X_test, y_train, y_test = fetch_mnist()

    # Add debugging information
    print("\nData shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Verify data integrity
    print("\nVerifying data integrity...")
    print(f"X_train range: [{X_train.min()}, {X_train.max()}]")
    print(f"y_train unique values: {np.unique(y_train)}")
    print(f"X_test range: [{X_test.min()}, {X_test.max()}]")
    print(f"y_test unique values: {np.unique(y_test)}")

    # Verify expected sizes
    assert len(X_train) == 60000, f"Expected 60000 training images, got {len(X_train)}"
    assert len(y_train) == 60000, f"Expected 60000 training labels, got {len(y_train)}"
    assert len(X_test) == 10000, f"Expected 10000 test images, got {len(X_test)}"
    assert len(y_test) == 10000, f"Expected 10000 test labels, got {len(y_test)}"

    batch_size = 32
    print(f"\nCreating DataLoader with batch_size={batch_size}")
    train_loader = DataLoader(X_train, y_train, batch_size=batch_size)

    print("\nFetching first batch...")
    images, labels = next(iter(train_loader))
    
    # Access shape through data attribute
    print(f"Batch shapes - images: {images.data.shape}, labels: {labels.data.shape}")

    plt.figure(figsize=(10, 2))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images.data[i][0], cmap='gray')
        plt.title(f'Label: {labels.data[i]}')
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()