#!/usr/bin/env python
"""
Script to compute EER and plot ROC curves for multiple models (wav2vec, whisper, MFCC).
Usage:
$: python evaluate_2021LA_roc.py PATH_TO_GROUNDTRUTH_DIR phase

-PATH_TO_GROUNDTRUTH_DIR: path to the directory that has the CM protocol
-phase: either progress, eval, or hidden_track

Example:
$: python evaluate_all2021LA_roc.py ./keys eval
"""

import sys
import os.path
import numpy as np
import pandas
import matplotlib.pyplot as plt
import eval_metric_LA as em

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

def plot_multiple_roc_curves(models_data, output_filename="roc_curve_comparison21LA.png"):
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
    plt.title('ROC Curve for all Models on ASVspoo2021 LA Dataset')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Save the plot
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"ROC curve comparison saved as '{output_filename}'")
    plt.show()
def evaluate_model_roc(score_file, cm_key_file, phase, model_name):
    """
    Evaluate a single model and return ROC data.
    """
    print(f"Evaluating {model_name}...")
    
    # Load metadata
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    
    # Load scores
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)
    
    # Merge data based on phase
    cm_scores = submission_scores.merge(cm_data[cm_data[7] == phase], left_on=0, right_on=1, how='inner')
    
    # Extract bonafide and spoof scores
    bona_cm = cm_scores[cm_scores[5] == 'bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[5] == 'spoof']['1_x'].values
    
    print(f"After merging: found {len(bona_cm)} bonafide and {len(spoof_cm)} spoof samples")
    
    if len(spoof_cm) == 0 or len(bona_cm) == 0:
        print(f"Error: No spoof or bonafide samples found for {model_name}.")
        exit(1)
    
    # Compute EER
    eer_cm, eer_threshold = em.compute_eer(bona_cm, spoof_cm)
    
    print(f"{model_name} EER: {eer_cm*100:.2f}% at threshold {eer_threshold:.4f}")
    
    return model_name, bona_cm, spoof_cm, eer_cm, eer_threshold

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_2021LA_roc.py PATH_TO_GROUNDTRUTH_DIR phase")
        exit(1)
    
    truth_dir = sys.argv[1]
    phase = sys.argv[2]
    
    cm_key_file = os.path.join(truth_dir, 'CM/trial_metadata.txt')
    
    # Paths to score files for each model
    wav2vec_score_file = "eval_CM_09_052_la_wav2vecnew.txt"
    whisper_score_file = "eval_CM_LA_whisper_1505.txt"
    mfcc_score_file = "eval_CM_LA_mfcc_16052nd.txt"
    
    # Evaluate each model
    models_data = []
    models_data.append(evaluate_model_roc(wav2vec_score_file, cm_key_file, phase, "Wav2Vec"))
    models_data.append(evaluate_model_roc(whisper_score_file, cm_key_file, phase, "Whisper"))
    models_data.append(evaluate_model_roc(mfcc_score_file, cm_key_file, phase, "MFCC"))
    
    # Plot all ROC curves
    plot_multiple_roc_curves(models_data)