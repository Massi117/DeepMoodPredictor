# This script run LOO validation on the CNN model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from models import MoodCNNClassifier
from datasets import COPEDataset
import datamanager


if __name__ == "__main__":

    # Load the data
    X, y = datamanager.load_data(cope_type='cope_diff')

    # Determine number of folds
    folds = len(y)
    print(f"Performing LOO validation on {folds} samples...")

    # Start LOO validation
    preds_list = np.array([])
    actual_list = np.array([])
    for i in range(folds):
        # Prepare training and validation data
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_val = X[i:i+1]
        y_val = y[i:i+1]

        # Convert to datasets
        train_dataset = COPEDataset(X_train, y_train, augmentation=False)
        val_dataset = COPEDataset(X_val, y_val)

        # Define data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Setup
        model = MoodCNNClassifier().cuda()
        optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        num_epochs = 25

        # Train
        for epoch in range(num_epochs):
            model.train()
            total_loss, correct = 0.0, 0

            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.cuda(), labels.cuda()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == torch.argmax(labels, dim=1)).sum().item()

            acc = correct / len(train_loader.dataset)
            print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}, Accuracy: {acc:.4f}")


        # Validate
        print(f"Validating on sample {i+1}/{folds}...")
        model.eval()
        for input, label in val_loader:
            input, label = input.cuda(), label.cuda()
            input = input.clone().detach().requires_grad_(True)
            output = model(input)
            print(f"Raw output logits: {output.detach().cpu().numpy()[0]}")
            pred = torch.argmin(outputs, dim=1) # Sign inversion for uknown reason
            actual = torch.argmax(labels, dim=1)



            score = output[0, actual]
            model.zero_grad()
            score.backward()

            saliency = input.grad.abs()
            saliency = saliency[0, 0] 



            print(f'Predicted: {pred.item()}   True: {actual.item()}')
            # Update overall lists
            preds_list = np.append(preds_list, pred.item())
            actual_list = np.append(actual_list, actual.item())

        print('LOO iteration complete.\n')

    # Overall evaluation
    print("LOO Validation Complete.")
    print("Calculating overall validation accuracy...")
    val_acc = np.sum(preds_list == actual_list) / len(actual_list)
    print(f"Validation Accuracy: {val_acc:.4f}")

    cm = confusion_matrix(actual_list, preds_list)
    matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['SD', 'WR'])
    plt.title("Confusion Matrix")
    matrix.plot(cmap=plt.cm.Blues).figure_.savefig("figures/loo_confusion_matrix.png")
    plt.close()