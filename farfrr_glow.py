#!/usr/bin/env python
from pathlib import Path
import sys, os.path
import numpy as np
import pandas
import eval_metrics_DF as em # Assuming this file exists and has compute_eer
from glob import glob

# Argument handling
if len(sys.argv) < 2:
    print("Usage: python evaluate_glow.py <score_file_path>")
    sys.exit(1)

submit_file = sys.argv[1]

# --- Configuration for metadata file ---
# Adjust this path if your 'tts_mydataset' directory is located elsewhere
# relative to where you run this script.
# Example: If evaluate_glow.py is in SSL_Anti-spoofing/ and tts_mydataset is also in SSL_Anti-spoofing/
truth_dir_base = Path('.') # Assumes tts_mydataset is in the current working directory
# Or, if evaluate_glow.py is in a specific folder and tts_mydataset is relative to that:
# truth_dir_base = Path(__file__).resolve().parent 
truth_dir = truth_dir_base / 'tts_mydataset' / 'glow_tts'
cm_key_file = truth_dir / 'merged_ALL1000_protocols.txt' # Ensure this filename is correct

def compute_far_only(spoof_scores, threshold):
    """
    Compute False Acceptance Rate (FAR) for spoof samples.
    FAR = Number of spoofs accepted as bonafide / Total number of spoofs.
    Assumes: Lower scores = more likely bonafide (accepted).
    """
    spoof_scores = np.array(spoof_scores)
    if len(spoof_scores) == 0:
        return 0.0
    
    false_accepts = np.sum(spoof_scores < threshold)
    total_spoofs = len(spoof_scores)
    
    far = false_accepts / total_spoofs if total_spoofs > 0 else 0.0
    return far

def compute_frr_only(bona_scores, threshold):
    """
    Compute False Rejection Rate (FRR) for bonafide samples.
    FRR = Number of bonafide samples rejected as spoof / Total number of bonafide samples.
    Assumes: Higher scores = more likely spoof (rejected).
    """
    bona_scores = np.array(bona_scores)
    if len(bona_scores) == 0:
        return 0.0

    false_rejects = np.sum(bona_scores >= threshold) 
    total_bona = len(bona_scores)
    
    frr = false_rejects / total_bona if total_bona > 0 else 0.0
    return frr

def compute_far_at_thresholds_spoof_only(spoof_scores):
    """
    Compute and print FAR at different thresholds (for spoof-only analysis).
    """
    spoof_scores = np.array(spoof_scores)
    if len(spoof_scores) == 0:
        print("No spoof scores to compute FAR table.")
        return []
        
    min_score = np.min(spoof_scores)
    max_score = np.max(spoof_scores)

    if min_score == max_score: # Handle case with all same scores
        thresholds = np.array([min_score - 0.1, min_score, min_score + 0.1])
    else:
        # Generate thresholds slightly beyond the min/max to see full range
        thresholds = np.linspace(min_score - 0.1 * abs(max_score - min_score), 
                                 max_score + 0.1 * abs(max_score - min_score), 
                                 30) # 30 steps
    
    print("\n False Acceptance Rate (FAR) at different thresholds (Spoof-Only Analysis):")
    print("Threshold | FAR (%) | Spoofs Accepted | Total Spoofs")
    print("-" * 60)
    
    results_table = []
    for th_val in thresholds:
        far = compute_far_only(spoof_scores, th_val)
        spoofs_accepted = np.sum(spoof_scores < th_val)
        
        print(f"{th_val:9.4f} | {far*100:7.2f} | {spoofs_accepted:15d} | {len(spoof_scores):12d}")
        results_table.append((th_val, far))
    
    return results_table

def print_detailed_far_frr_table(bona_scores, spoof_scores):
    """
    Compute and print FAR and FRR at different thresholds when both sample types are present.
    """
    bona_scores = np.array(bona_scores)
    spoof_scores = np.array(spoof_scores)

    if len(bona_scores) == 0 and len(spoof_scores) == 0:
        print("No bonafide or spoof scores to compute FAR/FRR table.")
        return

    all_scores_combined = []
    if len(bona_scores) > 0:
        all_scores_combined.extend(bona_scores)
    if len(spoof_scores) > 0:
        all_scores_combined.extend(spoof_scores)
    
    if not all_scores_combined:
        print("No scores available for threshold generation.")
        return

    min_s = np.min(all_scores_combined)
    max_s = np.max(all_scores_combined)

    if min_s == max_s: # Handle case with all same scores
         thresholds = np.array([min_s - 0.1, min_s, min_s + 0.1])
    else:
        # Generate thresholds slightly beyond the min/max to see full range
        thresholds = np.linspace(min_s - 0.1 * abs(max_s - min_s), 
                                 max_s + 0.1 * abs(max_s - min_s), 
                                 30) # 30 steps

    print("\n FAR and FRR at different thresholds:")
    header = "Threshold | FAR (%) | FRR (%) | Spoofs Acc. | Bonafide Rej. | Total Spoof | Total Bona"
    print(header)
    print("-" * len(header))

    for th_val in thresholds:
        far = compute_far_only(spoof_scores, th_val) if len(spoof_scores) > 0 else 0.0
        frr = compute_frr_only(bona_scores, th_val) if len(bona_scores) > 0 else 0.0
        
        spoofs_accepted = np.sum(spoof_scores < th_val) if len(spoof_scores) > 0 else 0
        bonafide_rejected = np.sum(bona_scores >= th_val) if len(bona_scores) > 0 else 0
        
        print(f"{th_val:9.4f} | {far*100:7.2f} | {frr*100:7.2f} | {spoofs_accepted:11d} | {bonafide_rejected:13d} | {len(spoof_scores):11d} | {len(bona_scores):10d}")


def evaluate_scores(score_file_path, metadata_key_file_path):
    """
    Main evaluation function. Reads scores and metadata, computes metrics.
    """
    print(f"Reading metadata from: {metadata_key_file_path}")
    if not os.path.isfile(metadata_key_file_path):
        print(f"Error: Metadata file not found at {metadata_key_file_path}")
        sys.exit(1)
        
    metadata_entries = []
    try:
        with open(metadata_key_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Expecting format like: - ID - - - label - eval
                if len(parts) >= 6: 
                    file_id = parts[1] 
                    label = parts[5]   
                    metadata_entries.append([file_id, label])
                # else:
                #     print(f"Skipping malformed line in metadata: {line.strip()}") # Optional: for debugging metadata
    except Exception as e:
        print(f"Error reading metadata file {metadata_key_file_path}: {e}")
        sys.exit(1)
    
    if not metadata_entries:
        print(f"Error: No valid entries parsed from metadata file {metadata_key_file_path}.")
        print("Ensure format is like: '- LA_0000 - - - bonafide - eval'")
        sys.exit(1)
    cm_data_df = pandas.DataFrame(metadata_entries, columns=["file_id", "label"])
    
    print(f"Reading scores from: {score_file_path}")
    if not os.path.isfile(score_file_path):
        print(f"Error: Score file not found at {score_file_path}")
        sys.exit(1)
    try:
        submission_scores_df = pandas.read_csv(score_file_path, sep=' ', header=None, names=["file_id", "score"], skipinitialspace=True)
    except pandas.errors.EmptyDataError:
        print(f"Error: Score file {score_file_path} is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading score file {score_file_path}: {e}")
        sys.exit(1)

    print(f"Found {len(cm_data_df)} entries in metadata file.")
    print(f"Found {len(submission_scores_df)} entries in score file.")
    
    if len(submission_scores_df) == 0:
        print("Error: Score file is empty or could not be parsed correctly.")
        sys.exit(1)

    # Basic check for expected number of trials, can be made more robust
    # if len(submission_scores_df) != len(cm_data_df):
    #     print(f'Warning: Submission has {len(submission_scores_df)} of {len(cm_data_df)} expected trials. Results might be partial or skewed.')

    if len(submission_scores_df.columns) > 2:
        print(f'Warning: Submission has {len(submission_scores_df.columns)} columns, expected 2. Check for extra spaces in score file.')
        # Attempt to use only the first two columns if more are present
        submission_scores_df = submission_scores_df.iloc[:, :2]
        submission_scores_df.columns = ["file_id", "score"]


    submission_scores_df["file_id"] = submission_scores_df["file_id"].astype(str)
    cm_data_df["file_id"] = cm_data_df["file_id"].astype(str)
            
    # Merge scores with metadata
    merged_scores_df = submission_scores_df.merge(cm_data_df, on="file_id", how='inner')
    
    if len(merged_scores_df) < len(submission_scores_df) :
        print(f"Warning: After merging, {len(merged_scores_df)} entries remain out of {len(submission_scores_df)} from score file.")
        print("This means some file_ids in your score file were not found in the metadata.")
    if len(merged_scores_df) < len(cm_data_df) :
        print(f"Warning: After merging, {len(merged_scores_df)} entries remain out of {len(cm_data_df)} from metadata file.")
        print("This means some file_ids in your metadata were not found in the score file.")

    if len(merged_scores_df) == 0:
        print("Error: No matching file_ids found between score file and metadata. Cannot proceed.")
        sys.exit(1)

    bona_scores_arr = merged_scores_df[merged_scores_df["label"] == 'bonafide']["score"].values
    spoof_scores_arr = merged_scores_df[merged_scores_df["label"] == 'spoof']["score"].values
    
    print(f"After merging: found {len(bona_scores_arr)} bonafide and {len(spoof_scores_arr)} spoof samples with scores.")
    
    if len(spoof_scores_arr) == 0 and len(bona_scores_arr) == 0:
        print("Error: No bonafide or spoof samples with scores found after processing. Cannot compute metrics.")
        sys.exit(1)
    
    final_metric_value = -1 # Default for cases where EER/FAR cannot be computed

    if len(bona_scores_arr) == 0:
        print("\nNo bonafide samples found. Computing FAR for spoof samples only.")
        if len(spoof_scores_arr) > 0:
            print(f"\nSpoof Score Statistics:")
            print(f"  Mean: {np.mean(spoof_scores_arr):.4f}, Std: {np.std(spoof_scores_arr):.4f}, Min: {np.min(spoof_scores_arr):.4f}, Max: {np.max(spoof_scores_arr):.4f}")
            
            far_results_table = compute_far_at_thresholds_spoof_only(spoof_scores_arr)
            
            # Optimal threshold logic (targeting a specific FAR, e.g., 5%)
            optimal_threshold_val = None
            target_far_rate = 0.05 
            min_difference = float('inf')
            
            for th_val, far_val in far_results_table:
                difference = abs(far_val - target_far_rate)
                if difference < min_difference:
                    min_difference = difference
                    optimal_threshold_val = th_val
            
            if optimal_threshold_val is not None:
                optimal_far_at_th = compute_far_only(spoof_scores_arr, optimal_threshold_val)
                print(f"\nOptimal Results (targeting ~{target_far_rate*100}% FAR):")
                print(f"  Threshold: {optimal_threshold_val:.4f}")
                print(f"  Actual FAR at this threshold: {optimal_far_at_th*100:.2f}%")
                final_metric_value = optimal_far_at_th
            else:
                print("\nCould not determine an optimal threshold for the target FAR from the generated thresholds.")
        else:
            print("No spoof samples to analyze for FAR.")
        
    else: # Bonafide samples exist
        if len(spoof_scores_arr) == 0:
            print("\nNo spoof samples found. Cannot compute EER or full FAR/FRR table.")
            # Optionally, you could print FRR table for bonafide samples here if desired
        else:
            # Both bonafide and spoof samples are present
            print_detailed_far_frr_table(bona_scores_arr, spoof_scores_arr)
            
            # EER calculation using eval_metrics_DF
            try:
                # Assuming em.compute_eer returns (eer, threshold)
                eer_cm_val, eer_threshold_val = em.compute_eer(bona_scores_arr, spoof_scores_arr) 
                print("\nEER: %.2f%% at threshold %.4f" % (100 * eer_cm_val, eer_threshold_val))
                final_metric_value = eer_cm_val
            except Exception as e:
                print(f"\nError computing EER with em.compute_eer: {e}")
                print("Please ensure 'eval_metrics_DF.py' and its 'compute_eer' function are correct.")
                print("Continuing without EER.")

    return final_metric_value

if __name__ == "__main__":
    # Check if files exist before calling the main evaluation function
    if not os.path.isfile(submit_file):
        print(f"Error: Score file '{submit_file}' does not exist.")
        sys.exit(1)
        
    if not os.path.isfile(cm_key_file):
        print(f"Error: Metadata keys file '{cm_key_file}' does not exist.")
        print(f"Please check the path: {os.path.abspath(cm_key_file)}")
        sys.exit(1)

    print(f"Starting evaluation with score file: {submit_file}")
    print(f"Using metadata key file: {cm_key_file}")
    
    result_metric = evaluate_scores(submit_file, cm_key_file)
    
    if result_metric != -1:
        print(f"\nScript finished. Final primary metric value: {result_metric*100:.2f}%")
    else:
        print("\nScript finished. Could not compute a primary metric (EER or target FAR).")