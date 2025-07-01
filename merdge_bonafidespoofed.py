import os
import shutil
from pathlib import Path

def merge_datasets_three_files(bonafide_protocols_file, bonafide_wav_dir, spoof_protocols_file, spoof_wav_dir,
                               output_ids_file, output_protocols_file, output_wav_dir):
    """
    Merge datasets into 3 separate files, adapting to observed 4-part split.
    """

    print("--- SCRIPT START ---")
    print("Checking input files and directories...")
    # (Input check code remains the same)
    all_inputs_valid = True
    if not os.path.exists(bonafide_protocols_file):
        print(f"ERROR: Bonafide protocols file NOT FOUND: {bonafide_protocols_file}")
        all_inputs_valid = False
    if not os.path.isdir(bonafide_wav_dir):
        print(f"ERROR: Bonafide WAV directory NOT FOUND or is not a directory: {bonafide_wav_dir}")
        all_inputs_valid = False
    if not os.path.exists(spoof_protocols_file):
        print(f"ERROR: Spoof protocols file NOT FOUND: {spoof_protocols_file}")
        all_inputs_valid = False
    if not os.path.isdir(spoof_wav_dir):
        print(f"ERROR: Spoof WAV directory NOT FOUND or is not a directory: {spoof_wav_dir}")
        all_inputs_valid = False

    if not all_inputs_valid:
        print("One or more input paths are invalid. Please check the paths in the 'if __name__ == \"__main__\":' block.")
        return
    print("All input files/directories seem to exist.")

    print(f"\n--- Bonafide WAV Directory Listing ({bonafide_wav_dir}) ---")
    try:
        bonafide_all_files = os.listdir(bonafide_wav_dir)
        wav_files_bonafide = [f for f in bonafide_all_files if f.lower().endswith('.wav')]
        print(f"  Total items in directory: {len(bonafide_all_files)}")
        print(f"  Found {len(wav_files_bonafide)} .wav files.")
        if wav_files_bonafide: print(f"  First 5 WAV files: {wav_files_bonafide[:5]}")
        else: print(f"  No .wav files found in {bonafide_wav_dir}")
    except Exception as e: print(f"  ERROR listing bonafide WAV directory: {e}")

    print(f"\n--- Spoof WAV Directory Listing ({spoof_wav_dir}) ---")
    try:
        spoof_all_files = os.listdir(spoof_wav_dir)
        wav_files_spoof = [f for f in spoof_all_files if f.lower().endswith('.wav')]
        print(f"  Total items in directory: {len(spoof_all_files)}")
        print(f"  Found {len(wav_files_spoof)} .wav files.")
        if wav_files_spoof: print(f"  First 5 WAV files: {wav_files_spoof[:5]}")
        else: print(f"  No .wav files found in {spoof_wav_dir}")
    except Exception as e: print(f"  ERROR listing spoof WAV directory: {e}")

    Path(output_wav_dir).mkdir(parents=True, exist_ok=True)
    ids_list = []
    protocols_list = []
    bonafide_processed_count = 0
    spoof_processed_count = 0

    # --- Process Bonafide Samples ---
    print(f"\n--- Processing Bonafide Samples from: {bonafide_protocols_file} ---")
    try:
        with open(bonafide_protocols_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('//'):
                    if line_num <= 10: print(f"  Line {line_num}: Skipping comment or empty line: '{line}'")
                    continue

                if line_num <= 10: print(f"  Line {line_num} (raw): '{line}'")
                parts = line.split(' - ')
                if line_num <= 10: print(f"    Split into {len(parts)} parts: {parts}")

                # MODIFIED PARSING based on observed 4-part split: ['- ID', '-', 'LABEL', 'EVAL']
                if len(parts) == 4:
                    raw_id_part = parts[0].strip()  # Expected: '- 4'
                    # parts[1] is observed as '-' and ignored
                    label_from_file = parts[2].strip() # Expected: 'bonafide'
                    # parts[3] is 'eval'

                    # Extract ID from raw_id_part (e.g., from '- 4' to '4')
                    old_id = raw_id_part.lstrip('-').strip()
                    
                    if not old_id:
                        if line_num <=10: print(f"    WARNING: Extracted Bonafide old_id is empty from '{raw_id_part}'")
                        continue
                    
                    if line_num <= 10: print(f"    Processing Bonafide ID: '{old_id}' (from '{raw_id_part}'), Label: '{label_from_file}'")
                    
                    new_id = f"bonafide_{old_id}"
                    ids_list.append(new_id)
                    # Ensure protocol line uses "bonafide" as label, consistent with file format
                    protocol_line = f"- {new_id} - - - bonafide - eval" 
                    protocols_list.append(protocol_line)

                    found_wav = False
                    possible_wav_names = [f"{old_id}.wav", f"{old_id.lower()}.wav"] # Bonafide WAVs are like '4.wav'
                    if line_num <= 10: print(f"      Attempting to find WAV for ID '{old_id}' in '{bonafide_wav_dir}'. Trying names: {possible_wav_names}")

                    for wav_name in possible_wav_names:
                        old_wav_path = os.path.join(bonafide_wav_dir, wav_name)
                        if os.path.exists(old_wav_path):
                            new_wav_path = os.path.join(output_wav_dir, f"{new_id}.wav")
                            try:
                                shutil.copy2(old_wav_path, new_wav_path)
                                if line_num <= 10: print(f"      ✓ COPIED: '{old_wav_path}' to '{new_wav_path}'")
                                bonafide_processed_count += 1
                                found_wav = True
                                break 
                            except Exception as e:
                                if line_num <= 10: print(f"      ✗ ERROR copying '{old_wav_path}': {e}")
                                break 
                    if not found_wav and line_num <= 10:
                        print(f"      ✗ WAV NOT FOUND for Bonafide ID '{old_id}' with tried names.")
                elif line_num <= 10:
                    print(f"    Skipping line {line_num} - expected 4 parts after split, got {len(parts)}: '{line}'")
    except Exception as e:
        print(f"  ERROR reading or processing bonafide protocols file: {e}")
    print(f"--- Bonafide processing: {bonafide_processed_count} WAVs copied ---")

    # --- Process Spoof Samples ---
    print(f"\n--- Processing Spoof Samples from: {spoof_protocols_file} ---")
    try:
        with open(spoof_protocols_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('//'):
                    if line_num <= 10: print(f"  Line {line_num}: Skipping comment or empty line: '{line}'")
                    continue

                if line_num <= 10: print(f"  Line {line_num} (raw): '{line}'")
                parts = line.split(' - ')
                if line_num <= 10: print(f"    Split into {len(parts)} parts: {parts}")

                # MODIFIED PARSING based on observed 4-part split: ['- ID', '-', 'LABEL', 'EVAL']
                if len(parts) == 4:
                    raw_id_part = parts[0].strip() # Expected: '- 0'
                    # parts[1] is observed as '-' and ignored
                    label = parts[2].strip()       # Expected: 'spoof'
                    # parts[3] is 'eval'

                    old_id = raw_id_part.lstrip('-').strip()

                    if not old_id:
                        if line_num <=10: print(f"    WARNING: Extracted Spoof old_id is empty from '{raw_id_part}'")
                        continue
                        
                    if line_num <= 10: print(f"    Processing Spoof ID: '{old_id}' (from '{raw_id_part}'), Label: '{label}'")
                    new_id = f"spoof_{old_id}"
                    ids_list.append(new_id)
                    protocol_line = f"- {new_id} - - - {label} - eval" 
                    protocols_list.append(protocol_line)

                    found_wav = False
                    # Spoof WAVs are like 'glow_tts_0.wav'
                    possible_wav_names = [f"glow_tts_{old_id}.wav", f"{old_id}.wav"] 
                    if line_num <= 10: print(f"      Attempting to find WAV for ID '{old_id}' in '{spoof_wav_dir}'. Trying names: {possible_wav_names}")
                    
                    for wav_name in possible_wav_names:
                        old_wav_path = os.path.join(spoof_wav_dir, wav_name)
                        if os.path.exists(old_wav_path):
                            new_wav_path = os.path.join(output_wav_dir, f"{new_id}.wav")
                            try:
                                shutil.copy2(old_wav_path, new_wav_path)
                                if line_num <= 10: print(f"      ✓ COPIED: '{old_wav_path}' to '{new_wav_path}'")
                                spoof_processed_count += 1
                                found_wav = True
                                break
                            except Exception as e:
                                if line_num <= 10: print(f"      ✗ ERROR copying '{old_wav_path}': {e}")
                                break 
                    if not found_wav and line_num <= 10:
                         print(f"      ✗ WAV NOT FOUND for Spoof ID '{old_id}' with tried names.")
                elif line_num <= 10:
                    print(f"    Skipping line {line_num} - expected 4 parts after split, got {len(parts)}: '{line}'")
    except Exception as e:
        print(f"  ERROR reading or processing spoof protocols file: {e}")
    print(f"--- Spoof processing: {spoof_processed_count} WAVs copied ---")

    # --- Write Output Files ---
    # (Writing output files code remains the same)
    print("\n--- Writing Output Files ---")
    try:
        with open(output_ids_file, 'w') as f:
            for sample_id in ids_list:
                f.write(sample_id + '\n')
        print(f"  IDs file written: {output_ids_file} ({len(ids_list)} entries)")
    except Exception as e: print(f"  ERROR writing IDs file '{output_ids_file}': {e}")

    try:
        with open(output_protocols_file, 'w') as f:
            for protocol_line in protocols_list:
                f.write(protocol_line + '\n')
        print(f"  Protocols file written: {output_protocols_file} ({len(protocols_list)} entries)")
    except Exception as e: print(f"  ERROR writing protocols file '{output_protocols_file}': {e}")
    
    print(f"\n--- MERGE SUMMARY ---")
    print(f"Bonafide WAVs copied: {bonafide_processed_count}")
    print(f"Spoof WAVs copied: {spoof_processed_count}")
    print(f"Total entries in IDs list: {len(ids_list)}")
    print(f"Total entries in Protocols list: {len(protocols_list)}")
    print(f"Output WAV directory: {output_wav_dir}")
    print(f"--- SCRIPT END ---")

# Example usage
if __name__ == "__main__":
    # !!! IMPORTANT: VERIFY THESE PATHS ARE ABSOLUTELY CORRECT FOR YOUR SYSTEM !!!
    bonafide_protocols_path = r"/home/a/angelikkd/GitProjects/SSL_Anti-spoofing/bonafide_only_in_the_wild/bonafide_samples_first_1000.txt"
    bonafide_wav_directory = r"/home/a/angelikkd/GitProjects/SSL_Anti-spoofing/bonafide_only_in_the_wild"
    
    spoof_protocols_path = r"/home/a/angelikkd/GitProjects/SSL_Anti-spoofing/tts_mydataset/glow_tts/glow_tts_metadatakeys.txt"
    spoof_wav_directory = r"/home/a/angelikkd/GitProjects/SSL_Anti-spoofing/tts_mydataset/glow_tts/wavs"
    
    output_ids_path = r"/home/a/angelikkd/GitProjects/SSL_Anti-spoofing/tts_mydataset/glow_tts/merged_ALL1000_ids.txt"
    output_protocols_path = r"/home/a/angelikkd/GitProjects/SSL_Anti-spoofing/tts_mydataset/glow_tts/merged_ALL1000_protocols.txt"
    output_wav_directory = r"/home/a/angelikkd/GitProjects/SSL_Anti-spoofing/tts_mydataset/glow_tts/merged_ALL1000_wavs" # Changed from your log for consistency

    merge_datasets_three_files(
        bonafide_protocols_path, bonafide_wav_directory,
        spoof_protocols_path, spoof_wav_directory,
        output_ids_path, output_protocols_path, output_wav_directory
    )

