#!/usr/bin/env python
from pathlib import Path
import sys, os.path
import numpy as np
import pandas
import eval_metrics_DF as em
from glob import glob

# Fix argument handling
if len(sys.argv) < 2:
    print("Usage: python evaluate_in_the_wild.py <score_file>")
    exit(1)

submit_file = sys.argv[1]
truth_dir = Path.cwd() / 'keys'
cm_key_file = truth_dir / 'metadata_keys.txt'

def eval_to_score_file(score_file, cm_key_file):
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
    
    if len(bona_cm) == 0 or len(spoof_cm) == 0:
        print("Error: One of the score arrays is empty.")
        exit(1)
        
    # Calculate EER
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

    _ = eval_to_score_file(submit_file, cm_key_file)