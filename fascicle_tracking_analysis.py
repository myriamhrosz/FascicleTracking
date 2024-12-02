import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy import stats

def load_mat_file(filepath):
    """Load a .mat file and return its contents"""
    try:
        data = sio.loadmat(filepath)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return None

def extract_measurements(data):
    """Extract fascicle length and pennation measurements"""
    try:
        if 'TrackingData' in data:
            tracking = data['TrackingData'][0, 0]
            region = tracking['Region'][0, 0]
            
            # Extract measurements
            fl = np.array(region['fas_length'][0, 0], dtype=float).flatten()
            pen = np.array(region['fas_pen'][0, 0], dtype=float).flatten()
            num_frames = int(tracking['NumFrames'][0, 0][0, 0])
            time = np.arange(num_frames) / 30.0  # Assuming 30 fps
            
            # Remove any NaN values
            valid = ~np.isnan(fl) & ~np.isnan(pen)
            fl = fl[valid]
            pen = pen[valid]
            time = time[valid]
            
            return fl, pen, time
        else:
            print("No TrackingData found in file")
            return None, None, None
    except Exception as e:
        print(f"Error extracting measurements: {str(e)}")
        return None, None, None

def plot_time_series_comparison(measurements_dict):
    """Create detailed time series plots comparing measurements"""
    # Separate corrected and uncorrected measurements
    corrected = {k: v for k, v in measurements_dict.items() if 'correction' in k or 'v2' in k}
    uncorrected = {k: v for k, v in measurements_dict.items() if 'correction' not in k and 'v2' not in k}
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 15))
    
    # Plot fascicle lengths
    ax1 = plt.subplot(3, 1, 1)
    for name, (fl, _, time) in uncorrected.items():
        if fl is not None and time is not None:
            plt.plot(time, fl, '--', label=f'{name} (uncorrected)', alpha=0.7)
    for name, (fl, _, time) in corrected.items():
        if fl is not None and time is not None:
            plt.plot(time, fl, '-', label=f'{name}', alpha=0.7)
    plt.title('Fascicle Length over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Length (mm)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Plot pennation angles
    ax2 = plt.subplot(3, 1, 2)
    for name, (_, pen, time) in uncorrected.items():
        if pen is not None and time is not None:
            plt.plot(time, np.degrees(pen), '--', label=f'{name} (uncorrected)', alpha=0.7)
    for name, (_, pen, time) in corrected.items():
        if pen is not None and time is not None:
            plt.plot(time, np.degrees(pen), '-', label=f'{name}', alpha=0.7)
    plt.title('Pennation Angle over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Plot relative changes
    ax3 = plt.subplot(3, 1, 3)
    for name, (fl, _, time) in measurements_dict.items():
        if fl is not None and time is not None and len(fl) > 0:
            # Calculate relative change from initial value
            if fl[0] != 0:  # Only calculate if first value is non-zero
                rel_change = (fl - fl[0]) / fl[0] * 100
                style = '-' if ('correction' in name or 'v2' in name) else '--'
                plt.plot(time, rel_change, style, label=f'{name}', alpha=0.7)
    plt.title('Relative Change in Fascicle Length')
    plt.xlabel('Time (s)')
    plt.ylabel('Change (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('fascicle_time_series.png', bbox_inches='tight', dpi=300)
    plt.close()

def analyze_statistics(measurements_dict):
    """Analyze statistical properties of the measurements"""
    print("\nStatistical Analysis:")
    
    # Separate corrected and uncorrected measurements
    corrected = {k: v for k, v in measurements_dict.items() if 'correction' in k or 'v2' in k}
    uncorrected = {k: v for k, v in measurements_dict.items() if 'correction' not in k and 'v2' not in k}
    
    # Calculate statistics for each group
    print("\nFascicle Length Statistics:")
    print("\nCorrected measurements:")
    for name, (fl, _, _) in corrected.items():
        if fl is not None and len(fl) > 0:
            print(f"\n{name}:")
            print(f"  Mean ± SD: {np.mean(fl):.2f} ± {np.std(fl):.2f} mm")
            print(f"  Range: {np.min(fl):.2f} - {np.max(fl):.2f} mm")
            print(f"  Coefficient of Variation: {(np.std(fl)/np.mean(fl))*100:.2f}%")
    
    print("\nUncorrected measurements:")
    for name, (fl, _, _) in uncorrected.items():
        if fl is not None and len(fl) > 0:
            print(f"\n{name}:")
            print(f"  Mean ± SD: {np.mean(fl):.2f} ± {np.std(fl):.2f} mm")
            print(f"  Range: {np.min(fl):.2f} - {np.max(fl):.2f} mm")
            print(f"  Coefficient of Variation: {(np.std(fl)/np.mean(fl))*100:.2f}%")
    
    print("\nPennation Angle Statistics:")
    print("\nCorrected measurements:")
    for name, (_, pen, _) in corrected.items():
        if pen is not None and len(pen) > 0:
            print(f"\n{name}:")
            print(f"  Mean ± SD: {np.degrees(np.mean(pen)):.2f} ± {np.degrees(np.std(pen)):.2f} degrees")
            print(f"  Range: {np.degrees(np.min(pen)):.2f} - {np.degrees(np.max(pen)):.2f} degrees")
            print(f"  Coefficient of Variation: {(np.std(pen)/np.mean(pen))*100:.2f}%")
    
    print("\nUncorrected measurements:")
    for name, (_, pen, _) in uncorrected.items():
        if pen is not None and len(pen) > 0:
            print(f"\n{name}:")
            print(f"  Mean ± SD: {np.degrees(np.mean(pen)):.2f} ± {np.degrees(np.std(pen)):.2f} degrees")
            print(f"  Range: {np.degrees(np.min(pen)):.2f} - {np.degrees(np.max(pen)):.2f} degrees")
            print(f"  Coefficient of Variation: {(np.std(pen)/np.mean(pen))*100:.2f}%")

def analyze_mat_files(directory):
    """Analyze all .mat files in the given directory"""
    # Get all .mat files in directory
    mat_files = list(Path(directory).glob("*.mat"))
    print(f"Found {len(mat_files)} .mat files")
    
    # Dictionary to store measurements
    measurements = {}
    
    for mat_file in mat_files:
        if mat_file.name == 'ultrasound_tracking_settings.mat':
            continue
            
        print(f"\nAnalyzing {mat_file.name}:")
        data = load_mat_file(mat_file)
        
        if data is not None:
            # Extract measurements
            fl, pen, time = extract_measurements(data)
            if fl is not None and len(fl) > 0:
                measurements[mat_file.stem] = (fl, pen, time)
                print(f"  Successfully extracted {len(fl)} frames of data")
    
    if measurements:
        # Create time series plots
        plot_time_series_comparison(measurements)
        print("\nTime series plots have been saved to 'fascicle_time_series.png'")
        
        # Perform statistical analysis
        analyze_statistics(measurements)

def main():
    # Current directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Starting analysis of .mat files...")
    analyze_mat_files(current_dir)

if __name__ == "__main__":
    main()