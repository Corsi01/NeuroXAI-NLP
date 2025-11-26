import torch
import torch.nn as nn
import optuna
import numpy as np
from scipy.stats import pearsonr

from train_utils import training

class OneHiddenMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
           # nn.BatchNorm1d(hidden_dim),
           # nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
    
def objective(x_train_list, y_train_list, x_test_list, y_test_list, trial):
    
    #  HYPERPARAMETERS GRID
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    decay = trial.suggest_float('decay', 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256, 512, 1024])
    batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048])
    dropout_rate = trial.suggest_float('dropout_rate', 0.4, 0.6)
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(f"  lr={lr:.2e}, decay={decay:.2e}, hidden={hidden_dim}, batch={batch_size}")
    print(f"{'='*60}")
    
    subjects = [1, 2, 3, 5]
    input_dim = x_train_list[0].shape[1]  
    output_dim = y_train_list[0].shape[1]  
    
    # model creation function
    def create_mlp():
        return OneHiddenMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout_rate=dropout_rate
        )
    
    try:
        predictions_list = training(
            decay=decay,
            features_train_list=x_train_list,
            fmri_train_list=y_train_list,
            features_val_list=x_test_list,
            fmri_val_list=y_test_list,
            model_class=create_mlp,
            lr=lr,  
            batch_size=batch_size  
        )
    except Exception as e:
        print(f"Training failed: {e}")
        return -1.0  
    
    all_correlations = []
    
    for subj_idx in range(len(subjects)):
        y_pred = predictions_list[subj_idx]
        y_true = y_test_list[subj_idx]
        
        voxel_correlations = []
        for voxel_idx in range(y_true.shape[1]):
            r, _ = pearsonr(y_true[:, voxel_idx], y_pred[:, voxel_idx])
            voxel_correlations.append(r)
        
        mean_r = np.mean(voxel_correlations)
        all_correlations.append(mean_r)
        #print(f"  Subject {subjects[subj_idx]}: mean r = {mean_r:.4f}")
    
    final_score = np.mean(all_correlations)
    print(f" Final score (mean r): {final_score:.4f}")
    
    return final_score