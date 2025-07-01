#!/usr/bin/env python
"""
Script to compute EER and plot ROC curves for multiple models (wav2vec, whisper, MFCC) on the Tacotron dataset.
Usage:
$: python evaluate_all_Tacotron_roc.py PATH_TO_SCORE_FILES

Example:
$: python evaluate_all_Tacotron_roc.py ./
"""

import sys
import os.path
import numpy as np
import pandas
import matplotlib.pyplot as plt
import eval_metrics_DF as em  
from pathlib import Path

def compute_roc_curve(bona_scores, spoof_scores):
    """
    Compute ROC curve data points (FPR, TPR) for all possible thresholds.
    """
    all_scores = np.concatenate([bona_scores, spoof_scores])
    thresholds = np.sort(np.unique(all_scores))
    
    fpr_list = []
    tpr_list = []
    
    for threshold in thresholds:
        tpr = np.sum(bona_scores >= threshold) / len(bona_scores)
        fpr = np.sum(spoof_scores >= threshold) / len(spoof_scores)
        
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    
    return np.array(fpr_list), np.array(tpr_list), thresholds

def plot_multiple_roc_curves(models_data, output_filename="roc_curve_allmodels_Tacotron.png"):
    """
    Plot ROC curves for multiple models on the same plot with customizable colors.
    
    Args:
        models_data: List of tuples (model_name, bona_scores, spoof_scores, eer_value, eer_threshold)
        output_filename: Name of the output file to save the plot.
    """
    plt.figure(figsize=(8, 6))
    
    # Define custom colors for each model
    model_colors = {
        "Wav2Vec": "mediumorchid",
        "Whisper": "forestgreen",
        "MFCC": "darkorange"
    }
    
    for model_name, bona_scores, spoof_scores, eer_value, eer_threshold in models_data:
        fpr, tpr, thresholds = compute_roc_curve(bona_scores, spoof_scores)
        
        # Get the color for the current model
        color = model_colors.get(model_name, "black")  # Default to black if model name is not in the dictionary
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color=color, linewidth=2, label=f'{model_name} ROC Curve')
        
        # Mark EER point
        eer_fpr = eer_value
        eer_tpr = 1 - eer_value
        plt.plot(eer_fpr, eer_tpr, 'o', color=color, markersize=6, label=f'{model_name} EER: {eer_value*100:.2f}% at threshold {eer_threshold:.4f}')
    
    # Plot diagonal line (random guess)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random Guess')
    
    # Formatting
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve for all Models on Tacotron Dataset')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Save the plot
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"ROC curve comparison saved as '{output_filename}'")
    plt.show()

def evaluate_model_roc(score_file, cm_key_file, model_name):
    """
    Evaluate a single model and return ROC data using Tacotron format.
    """
    print(f"Evaluating {model_name}...")
    print(f"Score file: {score_file}")
    print(f"Metadata file: {cm_key_file}")
    
    # Process metadata file specially - it has a unique format (same as evaluate_tacotron)
    print(f"Reading metadata from: {cm_key_file}")
    
    # Custom parsing for the metadata file format
    metadata_entries = []
    with open(cm_key_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:  # Format: - ID - - - label - eval
                file_id = parts[1]
                label = parts[5]
                metadata_entries.append([file_id, label])
    
    cm_data = pandas.DataFrame(metadata_entries, columns=["file_id", "label"])
    
    # Read submission scores
    print(f"Reading scores from: {score_file}")
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None, names=["file_id", "score"], skipinitialspace=True)
    
    # Check counts
    print(f"Found {len(cm_data)} entries in metadata file")
    print(f"Found {len(submission_scores)} entries in score file")
    
    if len(submission_scores) != len(cm_data):
        print('CHECK: submission has %d of %d expected trials.' % (len(submission_scores), len(cm_data)))
        exit(1)

    if len(submission_scores.columns) > 2:
        print('CHECK: submission has more columns (%d) than expected (2). Check for leading/ending blank spaces.' % len(submission_scores.columns))
        exit(1)
    
    # Ensure all columns are properly typed for merging
    submission_scores["file_id"] = submission_scores["file_id"].astype(str)
    cm_data["file_id"] = cm_data["file_id"].astype(str)
            
    # Perform the merge
    cm_scores = submission_scores.merge(cm_data, on="file_id", how='inner')
    
    # Extract bonafide and spoof scores
    bona_cm = cm_scores[cm_scores["label"] == 'bonafide']["score"].values
    spoof_cm = cm_scores[cm_scores["label"] == 'spoof']["score"].values
    
    print(f"After merging: found {len(bona_cm)} bonafide and {len(spoof_cm)} spoof samples")
    
    if len(bona_cm) == 0 or len(spoof_cm) == 0:
        print(f"Error: One of the score arrays is empty for {model_name}.")
        print(f"Bonafide samples: {len(bona_cm)}, Spoof samples: {len(spoof_cm)}")
        exit(1)
    
    # Compute EER
    eer_cm, eer_threshold = em.compute_eer(bona_cm, spoof_cm)
    
    print(f"{model_name} EER: {eer_cm*100:.2f}% at threshold {eer_threshold:.4f}")
    
    return model_name, bona_cm, spoof_cm, eer_cm, eer_threshold

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_all_Tacotron_roc.py PATH_TO_SCORE_FILES")
        exit(1)
    
    score_files_dir = sys.argv[1]
    
    # Tacotron metadata file path (same as in evaluate_tacotron)
    truth_dir = Path('tts_mydataset/tactron2dcc_tts/')
    cm_key_file = truth_dir / 'merged_ALL1000_protocols.txt'
    
    # Check if metadata file exists
    if not os.path.exists(cm_key_file):
        print(f"Error: Metadata file not found at {cm_key_file}")
        exit(1)
    
    # Paths to score files for each model (update these to match your Tacotron score files)
    wav2vec_score_file = os.path.join(score_files_dir, "eval_CM_Tacotron_wav2vec_13061.txt")  
    whisper_score_file = os.path.join(score_files_dir, "eval_CM_Tacotron_whisper_13061.txt")
    mfcc_score_file = os.path.join(score_files_dir, "eval_CM_Tacotron_mfcc_13061.txt")
    
    # Check if score files exist
    for score_file in [wav2vec_score_file, whisper_score_file, mfcc_score_file]:
        if not os.path.exists(score_file):
            print(f"Warning: Score file not found: {score_file}")
    
    # Evaluate each model
    models_data = []
    
    # Only evaluate models whose score files exist
    if os.path.exists(wav2vec_score_file):
        models_data.append(evaluate_model_roc(wav2vec_score_file, cm_key_file, "Wav2Vec"))
    
    if os.path.exists(whisper_score_file):
        models_data.append(evaluate_model_roc(whisper_score_file, cm_key_file, "Whisper"))
    
    if os.path.exists(mfcc_score_file):
        models_data.append(evaluate_model_roc(mfcc_score_file, cm_key_file, "MFCC"))
    
    if len(models_data) == 0:
        print("Error: No valid models found to evaluate")
        exit(1)
    
    # Plot all ROC curves
    plot_multiple_roc_curves(models_data)