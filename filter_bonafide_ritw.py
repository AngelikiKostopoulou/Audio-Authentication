# this scricpt filters the release in the wild (RITW) dataset to only include bonafide samples
import pathlib
import pandas as pd 

BASE_PATH = pathlib.Path(__file__).parent / 'release_in_the_wild'
OUTPUT_PATH = BASE_PATH.parent / 'bonafide_only_in_the_wild'
def filter_bonafide_ritw():
    # Load the metadata file
    metadata_file = BASE_PATH / 'metadata_keys.txt'
    metadata = pd.read_csv(metadata_file, sep=' ', header=None, dtype=str)

    # The label is in the 5th column (index 5)
    bonafide_samples = metadata[metadata[5] == 'bonafide']
    output_file = BASE_PATH / 'bonafide_samples_RITW_NEW.txt'
    # Save the bonafide samples to a new file
    bonafide_samples.to_csv(output_file, index=False, header=False)

    # Select only the file ID column (index 1)
    bonafide_ids = bonafide_samples[1]

    # Save only the bonafide file IDs to a new file, one per line
    output_file = BASE_PATH / 'bonafide_ids_RITW_NEW.txt'
    bonafide_ids.to_csv(output_file, index=False, header=False)

    print(f"Filtered {len(bonafide_ids)} bonafide file IDs and saved to {output_file}")

# copy the first 1000 of the bonafide samples to a new folder:
def copy_first_1000_bonafide():
    import shutil

    bonafide_ids = pd.read_csv(BASE_PATH / 'bonafide_ids_RITW_NEW.txt', header=None, dtype=str)
    bonafide_ids = bonafide_ids[0].tolist()[:1000]  # Get the first 1000 IDs
    bonafide_samples = pd.read_csv(BASE_PATH / 'bonafide_samples_RITW_NEW.txt', header=None, dtype=str)[:1000]

    # source_dir = BASE_PATH / 'release_in_the_wild'  # Assuming the original files are in this directory
    target_dir = OUTPUT_PATH

    target_dir.mkdir(parents=True, exist_ok=True)

    for file_id in bonafide_ids:
        source_file = BASE_PATH / f"{file_id}.wav"  # Assuming files are .wav format
        if source_file.exists():
            shutil.copy(source_file, target_dir / f"{file_id}.wav")
        else:
            print(f"File {source_file} does not exist.")
    
    print(f"Copied {len(bonafide_ids)} bonafide samples to {target_dir}")
    # Also copy a txt with the bonafide IDs for the first 1000 samples
    with open(target_dir / 'bonafide_ids_first_1000.txt', 'w') as f:
        for file_id in bonafide_ids:
            f.write(f"{file_id}\n")
    # Also copy the metadata file for the first 1000 samples
    bonafide_samples.to_csv(target_dir / 'bonafide_samples_first_1000.txt', index=False, header=False, sep=' ')


if __name__ == "__main__":
    filter_bonafide_ritw()
    copy_first_1000_bonafide()