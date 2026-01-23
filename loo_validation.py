# This script run LOO validation on the CNN model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import nibabel as nib
import random

from models import CNN3D
from datasets import COPEDataset
import datamanager


if __name__ == "__main__":

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    try: 
        name = torch.cuda.get_device_name(0)
        count = torch.cuda.device_count()
        print(f"Device count: {count}")
        print(f"Device name: {name}")
    except RuntimeError:
        print('No GPUs detected')

    # Set seed
    seed = 42
    torch.manual_seed(seed) # For reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Load the data
    X, y = datamanager.load_data(cope_type='cope_diff', balanced=True)#, mask_dir='masks/MVP_rois/outer_brain-thr50-2mm.nii.gz')
    
    # Generate a list of indices from 0 to the length of the lists
    indices = list(range(len(y)))

    # Shuffle the lists
    random.shuffle(indices)
    X = [X[i] for i in indices]
    y = [y[i] for i in indices]

    # Define image affine for saving saliency maps
    img = nib.load('masks/MVP_rois/HarvardOxford-sub-maxprob-thr50-2mm.nii.gz')
    affine_set = img.affine

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
        train_dataset = COPEDataset(X_train, y_train)
        val_dataset = COPEDataset(X_val, y_val)


        # Define data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Setup
        model = CNN3D(in_channels=1, num_classes=2).cuda()
        optimizer = optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        num_epochs = 15

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
            output = output.softmax(dim=1)
            pred = torch.argmin(output, dim=1)
            print(f"Raw model output: {output.detach().cpu().numpy()}")
            print(f"Predicted class index: {pred.item()}")
            if pred.item() == 1:
                pred_class = 'R'
            else:
                pred_class = 'NR'
            actual = torch.argmax(label, dim=1)
            if actual.item() == 1:
                actual_class = 'R'
            else:
                actual_class = 'NR'
            
            # Saliency map computation
            print("Computing saliency map...")
            score = output[0, actual]
            model.zero_grad()
            score.backward()
            saliency = input.grad.abs()
            saliency = saliency[0, 0].cpu()

            # Save saliency map as NIfTI
            saliency_nifti = nib.Nifti1Image(saliency.numpy(), affine=affine_set)
            nib.save(saliency_nifti, f'masks/saliency_maps/loo_saliency_sample_{i}.nii.gz')
            print("Saliency map saved.")
            
            print(f'Predicted: {pred_class}   True: {actual_class}')
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
    matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NR', 'R'])
    plt.title("Confusion Matrix")
    matrix.plot(cmap=plt.cm.Blues).figure_.savefig("figures/loo_confusion_matrix.png")
    plt.close()