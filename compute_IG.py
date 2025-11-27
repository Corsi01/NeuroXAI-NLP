import torch
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

# Define function to load data for IG computation
def load_ig_data(sub, features_train, features_test):
    
    fmri = load_fmri('data', sub)
    
    X_train, y_train, scaler = align_features_and_fmri_samples(features_train, fmri, excluded_samples_start=5,
                                                               excluded_samples_end=5, hrf_delay=3,
                                                               movies=["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05"])
    
    X_test, y_test = align_features_and_fmri_samples(features_test, fmri, excluded_samples_start=5,
                                                     excluded_samples_end=5, hrf_delay=3,
                                                     movies=["friends-s06"], fitted_scaler=scaler)
    
    
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    del X_train, y_train, X_test, y_test, fmri, scaler
    
    return X_train_torch, X_test_torch

# Define function to compute IG for a specific layer
def compute_ig_layer(layer):
    
    features_train, features_test = load_stimulus_features(method='gpt2', layer=f'language_layer_{layer}')
    layer_dict = dict()
    
    for roi in ['LH_IFGorb', 'LH_IFG', 'LH_MFG', 'LH_AntTemp', 'LH_PostTemp']:
        
        fim_per_subject = []
        for subj in subjects:
            
            model = load_model(subject=subj, layer=layer, zone=roi)
            X_train_subject, X_test_subject = load_ig_data(subj, features_train, features_test)
            
            X_train_subject, X_test_subject = X_train_subject.to(device), X_test_subject.to(device)
            baseline_mean_vector = torch.mean(X_train_subject, dim=0, keepdim=True).to(device)
            
            def macro_roi_output_wrapper(input_features):
                return model(input_features).mean(dim=1) 
        
            ig = IntegratedGradients(macro_roi_output_wrapper)
            
            batch_attributions = []
            num_test_samples = X_test_subject.size(0)
            
            for i in tqdm(range(0, num_test_samples, BATCH_SIZE)):
                batch_inputs = X_test_subject[i:i + BATCH_SIZE]
                batch_baselines = baseline_mean_vector.repeat(batch_inputs.size(0), 1)
        
                IG_batch = ig.attribute(
                    inputs=batch_inputs,
                    baselines=batch_baselines,
                    n_steps=N_STEPS)
        
                batch_attributions.append(IG_batch.detach().cpu())
        
                del IG_batch, batch_inputs, batch_baselines
                gc.collect() 
                
            IG_tensor = torch.cat(batch_attributions, dim=0)
    
            FIM_S = torch.mean(IG_tensor, dim=0) 
            fim_per_subject.append(FIM_S.detach().cpu().numpy())
    
            del IG_tensor, FIM_S, model, X_train_subject, X_test_subject
            gc.collect()
    
    FIM_final_vector = np.mean(np.array(fim_per_subject), axis=0)
    layer_dict[roi] = FIM_final_vector
    
    return layer_dict

def main():    
    for l in tqdm(range(1, 13), desc="Processing Layers"):
        try:
            lay_ig_results = compute_ig_layer(layer=l)
            all_igs[f'layer_{l}'] = lay_ig_results
        except Exception as e:
            print(f"\nLayer {l}. Error: {e}")
            
    output_filename = 'FIM_Results_All_Combinations.pkl'
    with open(output_filename, 'wb') as f:
        pickle.dump(all_igs, f)
       

if __name__ == '__main__':
    main()