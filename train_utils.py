import os
import numpy as np
import pandas as pd
import json
import h5py
import random
from tqdm import tqdm, trange
from sklearn.preprocessing import StandardScaler
import nibabel as nib
from nilearn import plotting, datasets
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
import gc


def load_fmri(root_data_dir, subject):
 
    fmri = {}

    ### Load the fMRI responses for Friends ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
    fmri_dir = os.path.join(root_data_dir,    #### removed 'algonauts_2025.competitrs'
        'fmri', f'sub-0{subject}', 'func', fmri_file)
    # Load the the fMRI responses
    fmri_friends = h5py.File(fmri_dir, 'r')
    for key, val in fmri_friends.items():
        fmri[str(key[13:])] = val[:].astype(np.float32)
    del fmri_friends

    ### Load the fMRI responses for Movie10 ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
    fmri_dir = os.path.join(root_data_dir,   #### removed 'algonauts_2025.competitrs'
        'fmri', f'sub-0{subject}', 'func', fmri_file)
    # Load the the fMRI responses
    fmri_movie10 = h5py.File(fmri_dir, 'r')
    for key, val in fmri_movie10.items():
        fmri[key[13:]] = val[:].astype(np.float32)
    del fmri_movie10
    # Average the fMRI responses across the two repeats for 'figures'
    keys_all = fmri.keys()
    figures_splits = 12
    for s in range(figures_splits):
        movie = 'figures' + format(s+1, '02')
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]
    # Average the fMRI responses across the two repeats for 'life'
    keys_all = fmri.keys()
    life_splits = 5
    for s in range(life_splits):
        movie = 'life' + format(s+1, '02')
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]

    return fmri


def load_stimulus_features(method, layer):
    
    stimuli_list = []
    train_season = [1, 2, 3, 4, 5]
    movie_splits = []
    chunks_per_movie = []
    for i in train_season:
        stimuli_list.append(f'data/features/friends_s{i}_{method}_embeddings.h5')
     
    for i, stim_dir in tqdm(enumerate(stimuli_list)):
    
        data = h5py.File(stim_dir, 'r')
        for m, movie in enumerate(data.keys()):
        
            if i == 0 and m == 0: 
                features_train = np.asarray(data[movie][layer])
                        
            else:
                features_train = np.append(features_train, np.asarray(data[movie][layer]), 0)
            
            chunks_per_movie.append(len(data[movie][layer]))
            movie_splits.append(movie)
    
    print(features_train.shape)
    #print(features_train.nansum())
    #features_train = np.nan_to_num(features_train, nan=0)
    #scaler = StandardScaler()
    #scaler.fit(features_train)
    #features_train = scaler.transform(features_train)
    
    train_dict = {}
    count = 0
    for m, movie in enumerate(movie_splits):
        chunks = chunks_per_movie[m]
        clean_key = movie.replace('friends_', '')
        train_dict[clean_key] = features_train[count:count+chunks]
        count += chunks
        
    del features_train
        
    movie_splits = []
    chunks_per_movie = []
    data_dir = f'data/features/friends_s6_{method}_embeddings.h5'
    data = h5py.File(data_dir, 'r')
    
    for m, movie in enumerate(data.keys()):
        if m == 0:
            features_test = np.asarray(data[movie][layer])  
        else:
            features_test = np.append(features_test, np.asarray(data[movie][layer]), 0)
          
        chunks_per_movie.append(len(data[movie][layer]))
        movie_splits.append(movie)
                    
    print(features_test.shape)
    print(np.count_nonzero(np.isnan(features_test)))

    #print(features_train.nansum())
    
    test_dict = {}
    count = 0
    for m, movie in enumerate(movie_splits):
        chunks = chunks_per_movie[m]
        clean_key = movie.replace('friends_', '')
        test_dict[clean_key] = features_test[count:count+chunks]
        count += chunks
        
    del features_test
    
    return train_dict, test_dict

def align_features_and_fmri_samples(features, fmri, excluded_samples_start,
                                   excluded_samples_end, hrf_delay, movies, 
                                   fitted_scaler=None):
    """
    Simplified alignment for single TR (no temporal windowing).
    Aligns features 1:1 with fMRI, then removes NaN samples.
    """
    
    aligned_features = []
    aligned_fmri = []
    
    ### Loop across movies ###
    for movie in movies:
        # Get movie ID
        if movie[:7] == 'friends':
            id = movie[8:]
        movie_splits = [key for key in fmri if id in key[:len(id)]]
        
        ### Loop over movie splits ###
        for split in movie_splits:
            # Extract fMRI (with HRF delay)
            fmri_split = fmri[split]
            fmri_split = fmri_split[excluded_samples_start + hrf_delay : -excluded_samples_end]
            
            # Extract features (1:1 alignment)
            features_split = features[split]
            # Match fMRI samples: start from excluded_samples_start, take same length as fMRI
            features_split = features_split[excluded_samples_start : excluded_samples_start + len(fmri_split)]
            
            # Append
            aligned_features.append(features_split)
            aligned_fmri.append(fmri_split)
    
    ### Stack ###
    aligned_features = np.vstack(aligned_features)  # (n_samples, 768)
    aligned_fmri = np.vstack(aligned_fmri)          # (n_samples, 1000)
    
    print(f"Before NaN removal: {aligned_features.shape}")
    
    ### Remove NaN samples ###
    # Identify valid samples (no NaN in features)
    valid_mask = ~np.isnan(aligned_features).any(axis=1)
    
    aligned_features = aligned_features[valid_mask]
    aligned_fmri = aligned_fmri[valid_mask]
    
    print(f"After NaN removal: {aligned_features.shape}")
    print(f"Removed {(~valid_mask).sum()} NaN samples ({(~valid_mask).sum()/len(valid_mask)*100:.1f}%)")
    
    ### Standardize ###
    if fitted_scaler is None:
        scaler = StandardScaler()
        aligned_features = scaler.fit_transform(aligned_features)
    else:
        aligned_features = fitted_scaler.transform(aligned_features)
    
    ### Extract Fedorenko ROI ###
    with open('data/fedorendo_mask_language/mapping_fedorenko.json', 'r') as f:
        mapping = json.load(f)
    
    aligned_fmri_dict = {}
    for roi_name, roi_ids in mapping.items():
        # Average over parcels for this ROI
        aligned_fmri_dict[roi_name] = aligned_fmri[:, roi_ids]
    
    ### Output ###
    if fitted_scaler is None:
        return aligned_features, aligned_fmri_dict, scaler
    else:
        return aligned_features, aligned_fmri_dict

def plot_correlations(
    predictions_list,
    true_fmri_list,
    layer,
    fedorenko_zone,
    output_dir='./results'):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ROI IDs
    with open('data/fedorendo_mask_language/mapping_fedorenko.json', 'r') as f:
        roi_data = json.load(f)
    roi_ids = roi_data[fedorenko_zone]
    
    n_subjects = len(predictions_list)
    n_voxels = predictions_list[0].shape[1]
    
    for i in range(n_subjects):
        assert predictions_list[i].shape == true_fmri_list[i].shape, f"Shape mismatch for subject {i+1}"
        assert predictions_list[i].shape[1] == n_voxels, f"Voxel count mismatch for subject {i+1}"
    
    print(f"Processing {n_subjects} subjects with {n_voxels} voxels each...")
    
    # Calculate correlations and MSE for each subject
    all_correlations = []
    all_p_values = []
    all_mse = []
    
    for subj_idx in range(n_subjects):
        correlations = []
        p_values_list = []
        mse_values = []
        
        for i in range(n_voxels):
            # Correlation
            r, p = pearsonr(true_fmri_list[subj_idx][:, i], predictions_list[subj_idx][:, i])
            correlations.append(r)
            p_values_list.append(p)
            
            # MSE
            mse = np.mean((true_fmri_list[subj_idx][:, i] - predictions_list[subj_idx][:, i]) ** 2)
            mse_values.append(mse)
        
        all_correlations.append(correlations)
        all_p_values.append(p_values_list)
        all_mse.append(mse_values)
        
        print(f"  Subject {subj_idx+1}: mean r = {np.mean(correlations):.4f}, mean MSE = {np.mean(mse_values):.4f}")
    
    # Calculate mean across subjects
    all_correlations = np.array(all_correlations)  # shape: (n_subjects, n_voxels)
    all_mse = np.array(all_mse)  # shape: (n_subjects, n_voxels)
    
    mean_correlations = np.mean(all_correlations, axis=0)  # shape: (n_voxels,)
    mean_mse = np.mean(all_mse, axis=0)  # shape: (n_voxels,)
    
    print(f"Mean across subjects: r = {np.mean(mean_correlations):.4f}, MSE = {np.mean(mean_mse):.4f}")
    
    # Create comprehensive DataFrame
    df_dict = {'schaefer_1000_id': roi_ids}
    
    # Add individual subject correlations, p-values, and MSE
    for subj_idx in range(n_subjects):
        df_dict[f'correlation_sub{subj_idx+1}'] = all_correlations[subj_idx]
        df_dict[f'p_value_sub{subj_idx+1}'] = all_p_values[subj_idx]
        df_dict[f'significant_sub{subj_idx+1}'] = [p < 0.05 for p in all_p_values[subj_idx]]
        df_dict[f'mse_sub{subj_idx+1}'] = all_mse[subj_idx]
    
    # Add mean correlation and MSE
    df_dict['correlation_mean'] = mean_correlations
    df_dict['correlation_std'] = np.std(all_correlations, axis=0)
    df_dict['mse_mean'] = mean_mse
    df_dict['mse_std'] = np.std(all_mse, axis=0)
    
    results_df = pd.DataFrame(df_dict)
    
    # Save CSV
    csv_path = output_dir / f'voxel_correlations_{fedorenko_zone}_layer{layer}_all_subjects.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Load atlas
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=1000, resolution_mm=1)
    atlas_labels = nib.load(atlas.maps)
    
    # Create volumes using MEAN correlations
    atlas_data = atlas_labels.get_fdata()
    
    # 1. Binary mask for ROIs
    roi_mask = np.zeros_like(atlas_data, dtype=np.float32)
    
    # 2. Mean correlation volume
    corr_volume = np.full_like(atlas_data, np.nan, dtype=np.float32)
    
    for filtered_idx, schaefer_id in enumerate(roi_ids):
        mask = atlas_data == schaefer_id
        roi_mask[mask] = 1
        corr_volume[mask] = mean_correlations[filtered_idx]
    
    # Create images
    roi_mask_img = nib.Nifti1Image(roi_mask, atlas_labels.affine)
    corr_img = nib.Nifti1Image(corr_volume, atlas_labels.affine)
    
    # Determine vmin/vmax for symmetric colorbar
    vmax = max(abs(mean_correlations.min()), abs(mean_correlations.max()))
    vmin = -vmax
    
    # Plot mean correlations
    fig = plt.figure(figsize=(16, 6))
    
    # Layer 1: ROI mask in gray
    display = plotting.plot_glass_brain(
        roi_mask_img,
        threshold=0.5,
        cmap='gray',
        colorbar=False,
        alpha=0.3,
        figure=fig,
        display_mode='lyrz'
    )
    
    # Layer 2: Mean correlations
    display.add_overlay(
        corr_img,
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        colorbar=True
    )
    
    fig.suptitle(f'Mean Encoding Model Performance (n={n_subjects})\n'
                 f'{fedorenko_zone} | Layer {layer} | '
                 f'r = {np.mean(mean_correlations):.4f}, MSE = {np.mean(mean_mse):.4f}', 
                 fontsize=14, y=0.95)
    
    fig_path = output_dir / f'correlation_map_{fedorenko_zone}_layer{layer}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {fig_path}")
    plt.close()
    
    return results_df


def train_model(X_train, Y_train, X_val, Y_val, model_class, decay=1e-3, lr=1e-4, 
                batch_size=128, num_epochs=20, plot_curve=False):
    """
    Train model with proper memory management
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = model_class()
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    
    # Prepare data tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    
    # DataLoaders with memory management
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # CRITICAL: avoid worker leak
        pin_memory=(device.type == 'cuda'),
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,  # CRITICAL: avoid worker leak
        pin_memory=(device.type == 'cuda')
    )

    # Track losses
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in trange(num_epochs, desc="Training Epochs"):
        # TRAINING
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in train_loader:
            # Move to device
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            # Zero gradients BEFORE forward pass
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            # Backward pass
            loss.backward()
            
            # Optional: gradient clipping (uncomment if needed)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item() * x_batch.size(0)
            
            # IMPORTANT: Free memory batch
            del x_batch, y_batch, y_pred, loss

        # Free GPU cache after training epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Calculate average epoch loss
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # VALIDATION
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():  # Disable gradient tracking
            for x_val, y_val in val_loader:
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)
                
                val_pred = model(x_val)
                loss = criterion(val_pred, y_val)
                val_loss += loss.item() * x_val.size(0)
                
                # Free memory batch
                del x_val, y_val, val_pred, loss

        # Free GPU cache after validation
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Calculate average validation loss
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

    # CRITICAL: Move model to CPU to free GPU memory
    model = model.cpu()

    # Plot training curve if requested
    if plot_curve:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # CRITICAL: Final cleanup
    del X_train, Y_train, X_val, Y_val
    del train_dataset, val_dataset, train_loader, val_loader
    del optimizer, criterion
    
    # Garbage collection and GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return model

def training(features_train_list, fmri_train_list, features_val_list, fmri_val_list, 
             model_class, decay=1e-3, lr=1e-4, batch_size=128, num_epochs=20):
    
    prediction_list = []
    subjects = [1, 2, 3, 5]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(len(subjects)):
        print(f"\n=== Training Subject {subjects[i]} ===")    
        
        features_train = features_train_list[i]
        fmri_train = fmri_train_list[i]
        features_val = features_val_list[i]
        fmri_val = fmri_val_list[i]
        
        # Train model (returns model on CPU)
        trained_model = train_model(
            X_train=features_train,
            Y_train=fmri_train,
            X_val=features_val,
            Y_val=fmri_val,
            model_class=model_class,
            decay=decay,
            lr=lr,
            batch_size=batch_size,
            num_epochs=num_epochs,
            plot_curve=False
        )
        
        # Move to GPU for prediction
        trained_model = trained_model.to(device)
        trained_model.eval()
        
        # Prediction with no gradients
        with torch.no_grad():
            X_test = torch.tensor(features_val, dtype=torch.float32).to(device)
            y_pred = trained_model(X_test)
            y_pred = y_pred.cpu().numpy()
        
        prediction_list.append(y_pred)
        del trained_model, X_test
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    return prediction_list