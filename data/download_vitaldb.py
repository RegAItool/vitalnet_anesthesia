#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VitalDB Data Downloader
========================
Download multi-modal anesthesia data from VitalDB database.

Author: VitalNet Team
License: MIT
"""

import vitaldb
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def download_vitaldb_data(output_dir='./vitaldb_data', n_cases=100, duration=300):
    """
    Download VitalDB data with multiple physiological signals.

    Parameters
    ----------
    output_dir : str
        Directory to save downloaded data
    n_cases : int
        Number of cases to download (default: 100)
    duration : int
        Duration of data to extract per case in seconds (default: 300)

    Returns
    -------
    case_ids : list
        List of successfully downloaded case IDs
    """

    print("=" * 60)
    print("VitalNet Data Downloader")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Find cases with essential monitoring data
    print("\n[1/4] Searching for cases with complete monitoring data...")
    hr_cases = vitaldb.find_cases(['Solar8000/HR'])
    print(f"   Found {len(hr_cases)} cases with heart rate monitoring")

    # Select cases to download
    selected_cases = hr_cases[:n_cases]
    print(f"\n[2/4] Selected {len(selected_cases)} cases for download")

    # Define tracks to download
    tracks = [
        'Solar8000/HR',          # Heart Rate
        'Solar8000/NIBP_SBP',    # Non-invasive Blood Pressure - Systolic
        'Solar8000/NIBP_DBP',    # Non-invasive Blood Pressure - Diastolic
        'Solar8000/NIBP_MBP',    # Non-invasive Blood Pressure - Mean
        'Solar8000/SPO2',        # Oxygen Saturation
        'Solar8000/BT',          # Body Temperature
        'BIS/BIS',               # Bispectral Index
        'Orchestra/RFTN20_CE',   # Remifentanil Effect-site Concentration
        'Orchestra/RFTN20_CP',   # Remifentanil Plasma Concentration
        'Orchestra/PPF20_CE',    # Propofol Effect-site Concentration
        'Orchestra/PPF20_CP',    # Propofol Plasma Concentration
    ]

    print("\n[3/4] Downloading data...")
    print(f"   Tracks: {len(tracks)} physiological parameters")
    print(f"   Duration: {duration} seconds per case")

    successful_cases = []

    for idx, caseid in enumerate(tqdm(selected_cases, desc="   Progress")):
        try:
            # Load vital signs data
            data = vitaldb.load_case(caseid, tracks, duration)

            if data is not None and len(data) > 0:
                # Save to CSV
                output_file = os.path.join(output_dir, f'case_{caseid}.csv')
                data.to_csv(output_file, index=False)
                successful_cases.append(caseid)

        except Exception as e:
            print(f"\n   Warning: Failed to download case {caseid}: {str(e)}")
            continue

    print(f"\n[4/4] Download completed!")
    print(f"   Successfully downloaded: {len(successful_cases)}/{len(selected_cases)} cases")
    print(f"   Data saved to: {output_dir}")

    # Save case list
    case_list_file = os.path.join(output_dir, 'case_list.txt')
    with open(case_list_file, 'w') as f:
        for caseid in successful_cases:
            f.write(f"{caseid}\n")

    print(f"   Case list saved to: {case_list_file}")

    return successful_cases


def get_available_tracks():
    """
    Get list of available tracks in VitalDB.

    Returns
    -------
    tracks : list
        List of available track names
    """
    print("Querying VitalDB for available tracks...")
    print("Note: This may take a few moments...")

    # Sample a few cases to get track information
    sample_cases = vitaldb.find_cases(['Solar8000/HR'])[:5]

    all_tracks = set()
    for caseid in sample_cases:
        try:
            case_tracks = vitaldb.get_case_tracks(caseid)
            all_tracks.update(case_tracks)
        except:
            continue

    tracks = sorted(list(all_tracks))
    print(f"Found {len(tracks)} unique tracks")

    return tracks


def download_waveform_data(caseid, output_dir='./vitaldb_waveforms', duration=60):
    """
    Download high-frequency waveform data (ECG, arterial pressure).

    Parameters
    ----------
    caseid : int
        VitalDB case ID
    output_dir : str
        Directory to save waveform data
    duration : int
        Duration in seconds (default: 60)

    Returns
    -------
    success : bool
        Whether download was successful
    """

    os.makedirs(output_dir, exist_ok=True)

    waveform_tracks = [
        'SNUADC/ECG_II',    # ECG Lead II
        'SNUADC/ART',       # Arterial Pressure Waveform
        'SNUADC/PLETH',     # Plethysmography
    ]

    print(f"Downloading waveform data for case {caseid}...")

    try:
        for track in waveform_tracks:
            data = vitaldb.load_vital(caseid, [track], duration)

            if data is not None and len(data) > 0:
                track_name = track.replace('/', '_')
                output_file = os.path.join(output_dir, f'case_{caseid}_{track_name}.npy')
                np.save(output_file, data)
                print(f"   ✓ Saved {track}: {len(data)} samples")
            else:
                print(f"   × {track}: No data available")

        return True

    except Exception as e:
        print(f"   Error: {str(e)}")
        return False


if __name__ == '__main__':
    # Example usage

    # Download standard vital signs data
    print("\n" + "="*60)
    print("Example 1: Download vital signs data")
    print("="*60)

    cases = download_vitaldb_data(
        output_dir='./vitaldb_data',
        n_cases=10,
        duration=300
    )

    # Download waveform data for first case
    if len(cases) > 0:
        print("\n" + "="*60)
        print("Example 2: Download waveform data")
        print("="*60)

        download_waveform_data(
            caseid=cases[0],
            output_dir='./vitaldb_waveforms',
            duration=60
        )
