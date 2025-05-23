
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from torch import nn

class Training:
    def __init__(self, model, training_data, test_data, epochs, batch_size, loss_fn, learning_rate):
        self.model = model
        self.training_data = training_data
        self.test_data = test_data
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.train_dataloader = DataLoader(self.training_data, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            for X, y in self.train_dataloader:
                # zero out gradients
                self.optimizer.zero_grad()
                # forward pass
                y_pred = self.model(X)
                # loss function
                loss = self.loss_fn(y_pred, y)
                # backpropagation
                loss.backward()
                # update weights
                self.optimizer.step()
                # calculate loss
                epoch_loss += loss.item()
                num_batches += 1
            # Calculate average loss for the epoch
            avg_loss = epoch_loss / num_batches
            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}')

    def test(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        # Disable gradient calculation for testing
        with torch.no_grad(): 
            for X, y in self.test_dataloader:
                # Forward pass
                y_pred = self.model(X)
                
                # Calculate loss
                test_loss += self.loss_fn(y_pred, y).item()
                
                # Get predictions
                _, predicted = torch.max(y_pred.data, 1)
                
                # Update counts
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        # Calculate metrics
        avg_loss = test_loss / len(self.test_dataloader)
        accuracy = 100 * correct / total
        
        print('\nTest Results:')
        print(f'Average Loss: {avg_loss:.4f}')
        print(f'Accuracy: {accuracy:.2f}% ({correct}/{total})')