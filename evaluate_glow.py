#!/usr/bin/env python
from pathlib import Path
import sys, os.path
import numpy as np
import pandas
import eval_metrics_DF as em
from glob import glob

# Fix argument handling
if len(sys.argv) < 2:
    print("Usage: python evaluate_glow.py <score_file>")
    exit(1)

submit_file = sys.argv[1]
truth_dir = Path('tts_mydataset/glow_tts/')
cm_key_file = truth_dir / 'merged_ALL1000_protocols.txt'

def compute_far_only(spoof_scores, threshold=-1.8407):
    """
    Compute False Acceptance Rate (FAR) for spoof samples
    FAR = Number of spoofs accepted as bonafide / Total number of spoofs
    
    Lower scores = more likely bonafide (accepted)
    Higher scores = more likely spoof (rejected)
    """
    spoof_scores = np.array(spoof_scores)
    
    # Count how many spoofs are incorrectly accepted (score < threshold)
    false_accepts = np.sum(spoof_scores < threshold)
    total_spoofs = len(spoof_scores)
    
    far = false_accepts / total_spoofs
    return far

def compute_far_at_thresholds(spoof_scores):
    """
    Compute FAR at different thresholds
    """
    spoof_scores = np.array(spoof_scores)
    #thresholds = np.linspace(0.0, 1.0, 21)  # 0.0, 0.05, 0.1, ..., 1.0
    min_score = np.min(spoof_scores)
    max_score = np.max(spoof_scores)
    thresholds = np.linspace(min_score - 0.1, max_score + 0.1, 30)
    
    print("\n False Acceptance Rate (FAR) at different thresholds:")
    print("Threshold | FAR (%) | Spoofs Accepted | Total Spoofs")
    print("-" * 55)
    
    results = []
    for threshold in thresholds:
        far = compute_far_only(spoof_scores, threshold)
        spoofs_accepted = np.sum(spoof_scores < threshold)
        
        print(f"{threshold:7.2f}   | {far*100:6.1f}  | {spoofs_accepted:13d}   | {len(spoof_scores):10d}")
        results.append((threshold, far))
    
    return results

def eval_to_score_file_far_only(score_file, cm_key_file):
    """
    Modified function to calculate FAR only
    """
    # Process metadata file specially - it has a unique format
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
    
    if len(spoof_cm) == 0:
        print("Error: No spoof samples found.")
        exit(1)
    
    # Calculate FAR only (even if no bonafide samples)
    if len(bona_cm) == 0:
        print("\n No bonafide samples found. Computing FAR for spoof samples only.")
        
        # Score statistics
        print(f"\n Spoof Score Statistics:")
        print(f"Mean: {np.mean(spoof_cm):.4f}")
        print(f"Std:  {np.std(spoof_cm):.4f}")
        print(f"Min:  {np.min(spoof_cm):.4f}")
        print(f"Max:  {np.max(spoof_cm):.4f}")
        
        # Calculate FAR at different thresholds
        far_results = compute_far_at_thresholds(spoof_cm)
        
        # Find optimal threshold (where FAR is around 5-10%)
        optimal_threshold = None
        target_far = 0.05  # 5% FAR
        min_diff = float('inf')
        
        for threshold, far in far_results:
            diff = abs(far - target_far)
            if diff < min_diff:
                min_diff = diff
                optimal_threshold = threshold
        
        optimal_far = compute_far_only(spoof_cm, optimal_threshold)
        
        print(f"\n Optimal Results:")
        print(f"Threshold for ~5% FAR: {optimal_threshold:.3f}")
        print(f"Actual FAR at this threshold: {optimal_far*100:.2f}%")
        
        # Additional analysis
        print(f"\n Additional Analysis:")
        high_confidence_rejects = np.sum(spoof_cm > 0.8) / len(spoof_cm) * 100
        low_confidence_accepts = np.sum(spoof_cm < 0.2) / len(spoof_cm) * 100
        
        print(f"High-confidence spoof detections (>0.8): {high_confidence_rejects:.1f}%")
        print(f"Low-confidence (might be mistaken for bonafide, <0.2): {low_confidence_accepts:.1f}%")
        
        return optimal_far
        
    else:
        # Original EER calculation if both classes exist
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
        out_data = "eer: %.2f\n" % (100*eer_cm)
        print(out_data)
        return eer_cm

if __name__ == "__main__":
    if not os.path.isfile(submit_file):
        print("%s doesn't exist" % (submit_file))
        exit(1)
        
    if not os.path.isfile(cm_key_file):
        print("Metadata keys file doesn't exist at: %s" % (cm_key_file))
        exit(1)

    result = eval_to_score_file_far_only(submit_file, cm_key_file)