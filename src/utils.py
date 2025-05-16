import numpy as np
import gzip
import requests
import os
from typing import Tuple, List
from .tensor import Tensor

class DataLoader:
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True):
        # Add validation to ensure X and y have the same length
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}")
        
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        n_samples = len(self.X)
        indices = list(range(n_samples))
        if self.shuffle:
            np.random.shuffle(indices)
            
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            # Ensure we don't exceed array bounds
            if batch_indices[-1] >= n_samples:
                batch_indices = batch_indices[:n_samples - i]
            yield (
                Tensor(self.X[batch_indices]),
                Tensor(self.y[batch_indices])
            )
            
    def __len__(self):
        n_samples = len(self.X)
        
        return (n_samples + self.batch_size - 1) // self.batch_size

def fetch_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download and load MNIST dataset"""
    def download(filename: str, url: str):
        if not os.path.exists('data'):
            os.makedirs('data')
        filepath = f'data/{filename}'
        
        # Check if file already exists and is valid
        if os.path.exists(filepath):
            try:
                with gzip.open(filepath, 'rb') as f:
                    f.read(1)  # Try to read a byte to verify it's a valid gzip file
                print(f"Using existing {filename}")
                return
            except:
                print(f"Existing {filename} is corrupted, downloading again...")
                os.remove(filepath)
        
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Verify we got a gzip file
            if not response.headers.get('content-type', '').startswith('application/x-gzip'):
                raise ValueError(f"Expected gzip file, got {response.headers.get('content-type')}")
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
                
            # Verify the downloaded file
            with gzip.open(filepath, 'rb') as f:
                f.read(1)
            print(f"Successfully downloaded {filename}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
            raise
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            raise

    # MNIST URLs
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"  # Alternative URL
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }

    # Download files
    for filename in files.values():
        download(filename, base_url + filename)

    # Load and process data
    def load_images(filename: str) -> np.ndarray:
        try:
            with gzip.open(f'data/{filename}', 'rb') as f:
                # Add debug print
                print(f"Loading images from {filename}")
                data = np.frombuffer(f.read(), np.uint8, offset=16)
                # Add shape debug
                print(f"Raw data shape before reshape: {data.shape}")
                reshaped = data.reshape(-1, 1, 28, 28) / 255.0
                print(f"Reshaped data shape: {reshaped.shape}")
                return reshaped
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            raise

    def load_labels(filename: str) -> np.ndarray:
        try:
            with gzip.open(f'data/{filename}', 'rb') as f:
                # Add debug print
                print(f"Loading labels from {filename}")
                data = np.frombuffer(f.read(), np.uint8, offset=8)
                # Add shape debug
                print(f"Labels shape: {data.shape}")
                return data
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            raise

    try:
        # Add debug prints for each file
        print("\nLoading training data...")
        X_train = load_images(files["train_images"])
        y_train = load_labels(files["train_labels"])
        
        print("\nLoading test data...")
        X_test = load_images(files["test_images"])
        y_test = load_labels(files["test_labels"])
        
        # Verify shapes before returning
        print("\nVerifying final shapes:")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Verify that we're not mixing up training and test data
        assert len(X_train) == len(y_train), f"Training data mismatch: {len(X_train)} images vs {len(y_train)} labels"
        assert len(X_test) == len(y_test), f"Test data mismatch: {len(X_test)} images vs {len(y_test)} labels"
        
        return X_train, X_test, y_train, y_test  # Note: Changed order to match expected output
        
    except Exception as e:
        print(f"Error loading MNIST dataset: {e}")
        raise
