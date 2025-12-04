import torch
import torch.nn as nn
import numpy as np
from captum.attr import IntegratedGradients
from tqdm import tqdm
import pickle
import gc

from train_utils import load_fmri, load_stimulus_features, align_features_and_fmri_samples


# Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
subjects = [1, 2, 3, 5]
BATCH_SIZE = 2048
N_STEPS = 80 
all_igs = dict()

# Define function to load trained model
def load_model(subject, layer, zone, device='cuda'):
    with open(f'results/layer{layer}/subject_{subject}_layer{layer}_zone{zone}_model_full.pkl', 'rb') as f:
        model = pickle.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model

# Define simple MLP model architecture
class OneHiddenMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Define function to load data for IG computation
def load_ig_data(sub, features_train, features_test, fmri=None):
    """
    Load and align fMRI data with features.
    If fmri is provided, use it; otherwise load from disk.
    """
    if fmri is None:
        fmri = load_fmri('data', sub)
    
    X_train, y_train, scaler = align_features_and_fmri_samples(
        features_train, fmri, 
        excluded_samples_start=5,
        excluded_samples_end=5, 
        hrf_delay=3,
        movies=["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05"]
    )
    
    X_test, y_test = align_features_and_fmri_samples(
        features_test, fmri, 
        excluded_samples_start=5,
        excluded_samples_end=5, 
        hrf_delay=3,
        movies=["friends-s06"], 
        fitted_scaler=scaler
    )
    
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    
    # Clean up
    del X_train, y_train, X_test, y_test, scaler
    gc.collect()
    
    return X_train_torch, X_test_torch

# Define function to compute IG for a specific layer
def compute_ig_layer(layer):
    """
    Optimized version: loads features and fMRI data once per layer,
    then reuses them across all ROIs.
    """
    print(f"\n{'='*60}")
    print(f"Processing Layer {layer}")
    print(f"{'='*60}")
    
    # 1. Load features ONCE per layer (not per ROI!)
    print(f"Loading features for layer {layer}...")
    features_train, features_test = load_stimulus_features(
        method='gpt2', 
        layer=f'language_layer_{layer}'
    )
    
    # 2. Load and align data ONCE per subject (not per ROI!)
    print(f"Loading and aligning data for all subjects...")
    subject_data = {}
    
    for subj in tqdm(subjects, desc="Loading subjects"):
        # Load fMRI data
        fmri = load_fmri('data', subj)
        
        # Align with features
        X_train, X_test = load_ig_data(subj, features_train, features_test, fmri)
        
        # Move to GPU and compute baseline
        X_train = X_train.to(device)
        X_test = X_test.to(device)
        baseline = torch.mean(X_train, dim=0, keepdim=True)
        
        # Store data for this subject
        subject_data[subj] = {
            'X_test': X_test,
            'baseline': baseline
        }
        
        # Free memory - we don't need X_train anymore
        del X_train, fmri
        gc.collect()
    
    print(f"Data loaded. Memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    
    # 3. Now loop over ROIs using pre-loaded data
    layer_dict = dict()
    
    for roi in tqdm(['LH_IFGorb', 'LH_IFG', 'LH_MFG', 'LH_AntTemp', 'LH_PostTemp'], 
                     desc=f"Processing ROIs for layer {layer}"):
        
        fim_per_subject = []
        
        for subj in subjects:
            
            # Load model for this subject-layer-ROI combination
            model = load_model(subject=subj, layer=layer, zone=roi)
            
            # Get pre-loaded data
            X_test_subject = subject_data[subj]['X_test']
            baseline_mean_vector = subject_data[subj]['baseline']
            
            # Define wrapper for IntegratedGradients
            def macro_roi_output_wrapper(input_features):
                return model(input_features).mean(dim=1)
        
            ig = IntegratedGradients(macro_roi_output_wrapper)
            
            # Compute IG in batches
            batch_attributions = []
            num_test_samples = X_test_subject.size(0)
            
            for i in range(0, num_test_samples, BATCH_SIZE):
                batch_inputs = X_test_subject[i:i + BATCH_SIZE]
                batch_baselines = baseline_mean_vector.repeat(batch_inputs.size(0), 1)
        
                IG_batch = ig.attribute(
                    inputs=batch_inputs,
                    baselines=batch_baselines,
                    n_steps=N_STEPS
                )
        
                batch_attributions.append(IG_batch.detach().cpu())
        
                del IG_batch, batch_inputs, batch_baselines
                gc.collect()
            
            # Concatenate all batches
            IG_tensor = torch.cat(batch_attributions, dim=0)
            
            # Compute Feature Importance Map for this subject
            FIM_S = torch.mean(IG_tensor, dim=0)
            fim_per_subject.append(FIM_S.detach().cpu().numpy())
    
            # Clean up
            del IG_tensor, FIM_S, model
            gc.collect()
        
        # Average across subjects (FIXED INDENTATION!)
        FIM_final_vector = np.mean(np.array(fim_per_subject), axis=0)
        layer_dict[roi] = FIM_final_vector
        
        print(f"  ✓ {roi}: FIM shape = {FIM_final_vector.shape}")
    
    # Clean up subject data before returning
    del subject_data, features_train, features_test
    gc.collect()
    torch.cuda.empty_cache()
    
    return layer_dict

def main():
    """
    Main function to compute Integrated Gradients for all layers and ROIs.
    """
  
    for l in tqdm(range(1, 13), desc="Processing All Layers"):
        try:
            lay_ig_results = compute_ig_layer(layer=l)
            all_igs[f'layer_{l}'] = lay_ig_results
            
            # Verify results
            print(f"\nLayer {l} completed successfully!")
            print(f"  ROIs saved: {list(lay_ig_results.keys())}")
            
        except Exception as e:
            print(f"\n ERROR in Layer {l}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    output_filename = 'FIM_Results_All_Combinations.pkl'
    print(f"\n{'='*60}")
    print(f"Saving results to {output_filename}...")
    
    with open(output_filename, 'wb') as f:
        pickle.dump(all_igs, f)
    
   
    print("SUMMARY")
    for layer_key, layer_data in all_igs.items():
        print(f"{layer_key}: {len(layer_data)} ROIs")
        for roi, fim in layer_data.items():
            print(f"  - {roi}: shape {fim.shape}")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
