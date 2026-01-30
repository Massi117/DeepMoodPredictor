# This script run LOO validation on the CNN model
# Takes in command line argument for random seed (default 0)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import nibabel as nib
import random
import sys
import csv
import time

from models import CNN3D
from datasets import COPEDataset
import datamanager


if __name__ == "__main__":

    # Start Timer
    start_time = time.perf_counter()

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

    # Check and set random seeds
    if len(sys.argv) == 1:      # no arguments
        seed = 0
        mask_path = None
        save_file = 'test'
    elif len(sys.argv) == 2:    # one argument
        seed = int(sys.argv[1])
        mask_path = None
        save_file = ''
    elif len(sys.argv) == 3:    # two arguments
        seed = int(sys.argv[1])
        mask_path = sys.argv[2]
        save_file = mask_path.replace('-mask.nii.gz', '')
        save_file = save_file.replace('masks/MVP_rois/', '')

    print(f"Using random seed: {seed}")
    print(f'Using mask path: {mask_path}')

    torch.manual_seed(seed) # For reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Load the data
    X, y, code_list = datamanager.load_data(cope_type='cope_diff', balanced=True, seed=seed, mask_dir=mask_path)
    
    # Generate a list of indices from 0 to the length of the lists
    indices = list(range(len(y)))

    # Shuffle the lists
    random.shuffle(indices)
    X = [X[i] for i in indices]
    y = [y[i] for i in indices]
    code_list = [code_list[i] for i in indices]

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
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False
        )

        # Setup
        torch.set_float32_matmul_precision('high')
        model = torch.compile(CNN3D(in_channels=1, num_classes=1)).cuda()
        optimizer = optim.RMSprop(model.parameters(), lr=1e-5)
        criterion = nn.BCEWithLogitsLoss()
        num_epochs = 15

        # Train
        for epoch in range(num_epochs):
            model.train()
            total_loss, correct = 0.0, 0

            scaler = torch.amp.GradScaler('cuda')

            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.detach()
                probs = torch.sigmoid(logits)                
                preds = torch.round(probs)
                correct += (preds == labels).sum().item()

            acc = correct / len(train_loader.dataset)
            print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}, Accuracy: {acc:.4f}")


        # Validate
        print(f"Validating on sample {i+1}/{folds}...")
        model.eval()
        for input, label in val_loader:
            input, label = input.cuda(), label.cuda()
            #input = input.clone().detach().requires_grad_(True)
            with torch.no_grad():
                logit = model(input)
            prob = torch.sigmoid(logit)                
            pred = torch.round(prob)
            pred = 1 - pred.item()
            
            '''
            # Saliency map computation
            print("Computing saliency map...")
            score = output[0, pred]
            model.zero_grad()
            score.backward()
            saliency = input.grad.abs()
            saliency = saliency[0, 0].cpu()

            # Save saliency map as NIfTI
            saliency_nifti = nib.Nifti1Image(saliency.numpy(), affine=affine_set)
            nib.save(saliency_nifti, f'masks/saliency_maps/loo_saliency_sample_{i}.nii.gz')
            print("Saliency map saved.")
            '''

            print(f"Raw model output: {prob.detach().cpu().numpy()}")
            print(f'Predicted: {pred}   True: {label.item()}')
            # Update overall lists
            preds_list = np.append(preds_list, pred)
            actual_list = np.append(actual_list, label.item())

        print('LOO iteration complete.\n')

    # Overall evaluation
    print("LOO Validation Complete.")
    print("Calculating overall validation accuracy...")
    val_acc = np.sum(preds_list == actual_list) / len(actual_list)
    print(f"Validation Accuracy: {val_acc:.4f}")
    

    # Append results to CSV
    new_row_values = [seed, val_acc]
    with open(f'data/accuracies_{save_file}.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the new list of values as a single row
        writer.writerow(new_row_values)
    '''
    cm = confusion_matrix(actual_list, preds_list)
    matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NR', 'R'])
    plt.title("Confusion Matrix")
    matrix.plot(cmap=plt.cm.Blues).figure_.savefig("figures/loo_confusion_matrix.png")
    plt.close()
    '''

    # Script runtime
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Script execution time: {elapsed_time:.4f} seconds")