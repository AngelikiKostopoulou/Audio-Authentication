"""
Most accurate audio comparison script using RMS Energy and Dynamic Range
Usage: python accurate_audio_compare.py <directory_path_or_file>
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import sys
import os
import glob
from pathlib import Path

def compute_accurate_metrics(audio, sr):
    """
    Compute the most reliable audio quality metrics for comparison
    """
    # Input validation
    if len(audio) == 0:
        raise ValueError("Audio array is empty")
    
    if np.all(audio == 0):
        print("⚠️  WARNING: Audio is completely silent - metrics may be unreliable")
    
    # 1. RMS Energy Analysis (most reliable)
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Add safety check for RMS values
    if np.any(rms <= 0):
        print(f"⚠️  WARNING: Found {np.sum(rms <= 0)} zero/negative RMS values - replacing with minimum positive value")
        min_positive_rms = np.min(rms[rms > 0]) if np.any(rms > 0) else 1e-10
        rms = np.maximum(rms, min_positive_rms)
    
    rms_db = 20 * np.log10(rms + 1e-10)
    
    # Validate dB values are reasonable
    if np.min(rms_db) < -120:
        print(f"⚠️  WARNING: Very low energy detected (min: {np.min(rms_db):.1f} dB) - possible audio corruption")
    
    # 2. Statistical Analysis of Energy
    rms_mean = np.mean(rms_db)
    rms_std = np.std(rms_db)
    rms_max = np.max(rms_db)
    rms_min = np.min(rms_db)
    
    # 3. Dynamic Range (most important for quality comparison)
    dynamic_range = rms_max - rms_min
    
    # Validate dynamic range is reasonable
    if dynamic_range > 150:
        print(f"⚠️  WARNING: Extremely high dynamic range ({dynamic_range:.1f} dB) - possible data error")
    
    # 4. Signal Activity (percentage of time above noise floor)
    noise_floor = np.percentile(rms_db, 20)  # Bottom 20% as noise estimate
    active_frames = np.sum(rms_db > (noise_floor + 6))  # 6dB above noise floor
    activity_ratio = active_frames / len(rms_db)
    
    # 5. Energy Consistency (lower = more consistent/cleaner)
    energy_consistency = rms_std / (rms_max - rms_min + 1e-10)
    
    # 6. Peak-to-Average Ratio
    peak_amplitude = np.max(np.abs(audio))
    avg_amplitude = np.mean(np.abs(audio))
    peak_to_avg_ratio = 20 * np.log10(peak_amplitude / (avg_amplitude + 1e-10))
    
    # Final validation summary
    print(f"📊 METRICS SUMMARY:")
    print(f"   RMS Range: {rms_min:.1f} to {rms_max:.1f} dB")
    print(f"   Dynamic Range: {dynamic_range:.1f} dB")
    print(f"   Noise Floor: {noise_floor:.1f} dB")
    print(f"   Activity Ratio: {activity_ratio:.1%}")
    
    return {
        'rms_mean_db': rms_mean,
        'rms_std_db': rms_std,
        'dynamic_range_db': dynamic_range,
        'activity_ratio': activity_ratio,
        'energy_consistency': energy_consistency,
        'peak_to_avg_ratio_db': peak_to_avg_ratio,
        'noise_floor_db': noise_floor,
        'rms_values': rms_db,
        'quality_score': calculate_quality_score(rms_mean, dynamic_range, activity_ratio, energy_consistency)
    }

def calculate_quality_score(rms_mean, dynamic_range, activity_ratio, energy_consistency):
    """
    Calculate overall quality score (0-100)
    """
    score = 0
    
    # RMS level contribution (0-25 points)
    if rms_mean > -20:
        score += 25
    elif rms_mean > -30:
        score += 20
    elif rms_mean > -40:
        score += 15
    else:
        score += 5
    
    # Dynamic range contribution (0-30 points)
    if dynamic_range > 30:
        score += 30
    elif dynamic_range > 20:
        score += 25
    elif dynamic_range > 15:
        score += 20
    elif dynamic_range > 10:
        score += 15
    else:
        score += 5
    
    # Activity ratio contribution (0-25 points)
    if activity_ratio > 0.7:
        score += 25
    elif activity_ratio > 0.5:
        score += 20
    elif activity_ratio > 0.3:
        score += 15
    else:
        score += 10
    
    # Energy consistency contribution (0-20 points) - lower is better
    if energy_consistency < 0.1:
        score += 20
    elif energy_consistency < 0.2:
        score += 15
    elif energy_consistency < 0.3:
        score += 10
    else:
        score += 5
    
    return min(score, 100)

def find_audio_files(directory, extensions=['.wav', '.flac', '.WAV', '.FLAC']):
    """
    Find all audio files in directory
    """
    audio_files = []
    
    if not os.path.isdir(directory):
        return audio_files
    
    for ext in extensions:
        pattern = os.path.join(directory, f"*{ext}")
        audio_files.extend(glob.glob(pattern))
    
    return sorted(audio_files)

def select_audio_file(directory):
    """
    Let user select an audio file from directory
    """
    audio_files = find_audio_files(directory)
    
    if not audio_files:
        print(f"No audio files found in directory: {directory}")
        return None
    
    print(f"\nFound {len(audio_files)} audio files:")
    print("-" * 80)
    
    for i, audio_file in enumerate(audio_files, 1):
        filename = os.path.basename(audio_file)
        file_size = os.path.getsize(audio_file) / (1024 * 1024)  # MB
        print(f"{i:2d}. {filename:<50} ({file_size:.2f} MB)")
    
    print("-" * 80)
    
    while True:
        try:
            choice = input(f"\nSelect file (1-{len(audio_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(audio_files):
                return audio_files[choice_num - 1]
            else:
                print(f"Please enter a number between 1 and {len(audio_files)}")
                
        except ValueError:
            print("Please enter a valid number or 'q' to quit")

def load_audio_file(audio_file):
    """
    Load audio file with fallback methods and diagnostic checks
    """
    try:
        audio, sr = librosa.load(audio_file, sr=None)
        
        # DIAGNOSTIC CHECKS - Add detailed audio validation
        print(f"\n=== AUDIO FILE DIAGNOSTICS ===")
        print(f"Sample rate: {sr} Hz")
        print(f"Duration: {len(audio)/sr:.2f} seconds")
        print(f"Total samples: {len(audio)}")
        print(f"Data type: {audio.dtype}")
        print(f"Audio range: [{np.min(audio):.6f}, {np.max(audio):.6f}]")
        print(f"Audio mean: {np.mean(audio):.6f}")
        print(f"Audio std: {np.std(audio):.6f}")
        print(f"Non-zero samples: {np.count_nonzero(audio)} ({np.count_nonzero(audio)/len(audio)*100:.1f}%)")
        
        # Check for problematic conditions
        if np.all(audio == 0):
            print("⚠️  WARNING: Audio file contains only zeros (silent)")
        elif np.max(np.abs(audio)) < 1e-6:
            print("⚠️  WARNING: Audio levels extremely low (possible corruption)")
        elif np.isnan(audio).any():
            print("⚠️  ERROR: Audio contains NaN values")
        elif np.isinf(audio).any():
            print("⚠️  ERROR: Audio contains infinite values")
        else:
            print("✓ Audio data appears normal")
            
        print(f"===============================\n")
        
        return audio, sr
    except:
        try:
            audio, sr = sf.read(audio_file)
            
            # Same diagnostics for soundfile loading
            print(f"\n=== AUDIO FILE DIAGNOSTICS (soundfile) ===")
            print(f"Sample rate: {sr} Hz")
            print(f"Duration: {len(audio)/sr:.2f} seconds")
            print(f"Total samples: {len(audio)}")
            print(f"Data type: {audio.dtype}")
            print(f"Audio range: [{np.min(audio):.6f}, {np.max(audio):.6f}]")
            print(f"Audio mean: {np.mean(audio):.6f}")
            print(f"Audio std: {np.std(audio):.6f}")
            print(f"Non-zero samples: {np.count_nonzero(audio)} ({np.count_nonzero(audio)/len(audio)*100:.1f}%)")
            
            if np.all(audio == 0):
                print("⚠️  WARNING: Audio file contains only zeros (silent)")
            elif np.max(np.abs(audio)) < 1e-6:
                print("⚠️  WARNING: Audio levels extremely low (possible corruption)")
            elif np.isnan(audio).any():
                print("⚠️  ERROR: Audio contains NaN values")
            elif np.isinf(audio).any():
                print("⚠️  ERROR: Audio contains infinite values")
            else:
                print("✓ Audio data appears normal")
                
            print(f"==========================================\n")
            
            return audio, sr
        except Exception as e:
            raise Exception(f"Could not load audio file: {e}")

def plot_accurate_analysis(audio_file):
    """
    Create focused, accurate analysis plot
    """
    # Load audio
    try:
        audio, sr = load_audio_file(audio_file)
        print(f"Loaded: {len(audio)} samples, {sr} Hz, {len(audio)/sr:.2f} seconds")
    except Exception as e:
        print(f"Error loading {audio_file}: {e}")
        return None
    
    # Compute accurate metrics
    metrics = compute_accurate_metrics(audio, sr)
    
    # Create time axes
    time = np.linspace(0, len(audio)/sr, len(audio))
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    times_rms = librosa.frames_to_time(np.arange(len(metrics['rms_values'])), sr=sr, hop_length=hop_length)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Waveform with RMS overlay
    plt.subplot(2, 3, 1)
    plt.plot(time, audio, alpha=0.6, linewidth=0.5, color='blue', label='Waveform')
    
    # Overlay RMS energy
    rms_amplitude = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_scaled = rms_amplitude * np.max(np.abs(audio)) / np.max(rms_amplitude)
    plt.plot(times_rms, rms_scaled, color='red', linewidth=2, label='RMS Energy')
    plt.plot(times_rms, -rms_scaled, color='red', linewidth=2, alpha=0.7)
    
    plt.title(f'Waveform + RMS - {os.path.basename(audio_file)}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. RMS Energy in dB (most important plot)
    plt.subplot(2, 3, 2)
    plt.plot(times_rms, metrics['rms_values'], color='green', linewidth=2)
    plt.axhline(y=metrics['noise_floor_db'], color='red', linestyle='--', alpha=0.7, label='Noise Floor')
    plt.axhline(y=metrics['rms_mean_db'], color='orange', linestyle='--', alpha=0.7, label='Mean Level')
    plt.title('RMS Energy over Time (Most Important)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Energy (dB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Energy Distribution Histogram
    plt.subplot(2, 3, 3)
    plt.hist(metrics['rms_values'], bins=30, alpha=0.7, color='purple', density=True)
    plt.axvline(x=metrics['noise_floor_db'], color='red', linestyle='--', label='Noise Floor')
    plt.axvline(x=metrics['rms_mean_db'], color='orange', linestyle='--', label='Mean Level')
    plt.title('Energy Distribution')
    plt.xlabel('Energy (dB)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Spectrogram (for visual reference)
    plt.subplot(2, 3, 4)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (Reference)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    
    # 5. Quality Metrics Comparison
    plt.subplot(2, 3, 5)
    metric_names = ['Quality\nScore', 'Dynamic\nRange', 'Activity\nRatio', 'Energy\nConsistency']
    metric_values = [
        metrics['quality_score'],
        metrics['dynamic_range_db'],
        metrics['activity_ratio'] * 100,  # Convert to percentage
        (1 - metrics['energy_consistency']) * 100  # Invert so higher is better
    ]
    
    colors = ['gold', 'green', 'blue', 'orange']
    bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.7)
    plt.title('Quality Metrics Comparison')
    plt.ylabel('Score/Value')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Detailed Metrics Text
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Quality assessment
    if metrics['quality_score'] >= 80:
        quality_level = "EXCELLENT"
    elif metrics['quality_score'] >= 65:
        quality_level = "GOOD"
    elif metrics['quality_score'] >= 50:
        quality_level = "FAIR"
    elif metrics['quality_score'] >= 35:
        quality_level = "POOR"
    else:
        quality_level = "VERY POOR"
    
    metrics_text = f"""
ACCURATE AUDIO METRICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: {os.path.basename(audio_file)}

KEY METRICS (for comparison):
Quality Score:     {metrics['quality_score']:.1f}/100
Mean RMS Level:    {metrics['rms_mean_db']:.1f} dB
Dynamic Range:     {metrics['dynamic_range_db']:.1f} dB
Activity Ratio:    {metrics['activity_ratio']:.1%}
Energy Consistency: {metrics['energy_consistency']:.3f}
Peak-to-Avg Ratio: {metrics['peak_to_avg_ratio_db']:.1f} dB
Noise Floor:       {metrics['noise_floor_db']:.1f} dB

ASSESSMENT: {quality_level}

COMPARISON GUIDE:
• Higher Quality Score = Better
• Higher Dynamic Range = Better  
• Higher Activity Ratio = More Content
• Lower Energy Consistency = Cleaner
• Compare these values between files
"""
    
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_file = f"{base_name}_accurate_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Analysis saved as: {output_file}")
    plt.show()
    
    return metrics

def main():
    if len(sys.argv) != 2:
        print("Usage: python accurate_audio_compare.py <directory_path_or_audio_file>")
        print("Examples:")
        print("  python accurate_audio_compare.py /path/to/audio/directory")
        print("  python accurate_audio_compare.py /path/to/audio.wav")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if not os.path.exists(path):
        print(f"Error: Path '{path}' does not exist")
        sys.exit(1)
    
    # Handle single file
    if os.path.isfile(path):
        print(f"Accurate Audio Analysis - Single File")
        print(f"File: {os.path.abspath(path)}")
        
        metrics = plot_accurate_analysis(path)
        if metrics:
            print(f"\n" + "="*60)
            print("COMPARISON METRICS:")
            print("="*60)
            print(f"Quality Score:      {metrics['quality_score']:.1f}/100")
            print(f"Mean RMS Level:     {metrics['rms_mean_db']:.1f} dB")
            print(f"Dynamic Range:      {metrics['dynamic_range_db']:.1f} dB")
            print(f"Activity Ratio:     {metrics['activity_ratio']:.1%}")
            print(f"Energy Consistency: {metrics['energy_consistency']:.3f}")
        return
    
    # Handle directory
    if not os.path.isdir(path):
        print(f"Error: '{path}' is not a valid directory")
        sys.exit(1)
    
    print(f"Accurate Audio Analysis - Directory Mode")
    print(f"Directory: {os.path.abspath(path)}")
    
    # Store results for comparison
    all_results = {}
    
    while True:
        selected_file = select_audio_file(path)
        
        if selected_file is None:
            break
        
        metrics = plot_accurate_analysis(selected_file)
        if metrics:
            filename = os.path.basename(selected_file)
            all_results[filename] = metrics
            
            print(f"\n" + "="*60)
            print("COMPARISON METRICS:")
            print("="*60)
            print(f"Quality Score:      {metrics['quality_score']:.1f}/100")
            print(f"Mean RMS Level:     {metrics['rms_mean_db']:.1f} dB")
            print(f"Dynamic Range:      {metrics['dynamic_range_db']:.1f} dB")
            print(f"Activity Ratio:     {metrics['activity_ratio']:.1%}")
            print(f"Energy Consistency: {metrics['energy_consistency']:.3f}")
        
        if len(all_results) > 1:
            print(f"\n" + "="*60)
            print("COMPARISON WITH PREVIOUS FILES:")
            print("="*60)
            print(f"{'Filename':<25} {'Quality':<8} {'RMS':<8} {'Dyn.Range':<10} {'Activity':<9}")
            print("-" * 60)
            
            for fname, result in all_results.items():
                print(f"{fname[:24]:<25} {result['quality_score']:7.1f} {result['rms_mean_db']:7.1f} "
                      f"{result['dynamic_range_db']:9.1f} {result['activity_ratio']:8.1%}")
        
        print("\n" + "="*60)
        continue_choice = input("Analyze another file? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()