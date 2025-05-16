import sys
import os 
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import fetch_mnist, DataLoader
from src.mnist_classifier import MNISTClassifier
from src.optim import Adam
from src.loss import CrossEntropyLoss


def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.data
        pred = output.data.argmax(axis = 1)
        correct += (pred == target.data).sum()
        total += target.data.shape[0]
        
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}: Loss = {loss.data:.4f}, '
                  f'Accuracy = {100. * correct / total:.2f}%')
            
    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        
        total_loss += loss.data
        pred = output.data.argmax(axis = 1)
        correct += (pred == target.data).sum()
        total += target.data.shape[0]
        
    return total_loss / len(test_loader), 100. * correct / total

def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Test Accuracy')
    
    plt.tight_layout()
    plt.show()

def main():
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    epochs = 5
    
    print("Loading MNIST dataset...")
    X_train, X_test, y_train, y_test = fetch_mnist()
    
    # Create data loaders
    train_loader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(X_test, y_test, batch_size=batch_size)
    
    # Initialize model, optimizer, and loss function
    model = MNISTClassifier()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    
    # Training history
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    print("\nStarting training...")
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        epoch_time = time.time() - start_time
        print(f'\nEpoch {epoch + 1}/{epochs} ({epoch_time:.2f}s):')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    # Plot training history
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    
    # Save the model (optional)
    # model.save('mnist_model.pth')

if __name__ == "__main__":
    main()