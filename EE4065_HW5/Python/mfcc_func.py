"""
MFCC Feature Extraction Function for Keyword Spotting
EE4065 Homework 5 - Q1: Keyword Spotting from Audio Signals
Based on Section 12.8 of the textbook

This module extracts MFCC (Mel-Frequency Cepstral Coefficients) features from audio files
using CMSIS-DSP library functions that are compatible with STM32 implementation.
"""

import os
import numpy as np
from scipy.io import wavfile
import cmsisdsp as dsp
import cmsisdsp.mfcc as mfcc
from cmsisdsp.datatype import F32


def create_mfcc_features(recordings_list, FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs, window):
    """
    Extract MFCC features from a list of audio recordings.
    
    Parameters:
    -----------
    recordings_list : list
        List of paths to WAV audio files
    FFTSize : int
        Size of the FFT window (typically 1024)
    sample_rate : int
        Audio sample rate in Hz (typically 8000)
    numOfMelFilters : int
        Number of Mel filter banks (typically 20)
    numOfDctOutputs : int
        Number of DCT outputs/MFCC coefficients (typically 13)
    window : ndarray
        Window function to apply (e.g., Hamming window)
    
    Returns:
    --------
    mfcc_features : ndarray
        MFCC features array of shape (num_samples, numOfDctOutputs * 2)
    labels : ndarray
        Labels for each sample (digit 0-9)
    """
    # Mel filter bank parameters
    freq_min = 20
    freq_high = sample_rate / 2
    
    # Create Mel filter bank matrix using CMSIS-DSP
    filtLen, filtPos, packedFilters = mfcc.melFilterMatrix(
        F32, freq_min, freq_high, numOfMelFilters, sample_rate, FFTSize
    )
    
    # Create DCT matrix for final MFCC computation
    dctMatrixFilters = mfcc.dctMatrix(F32, numOfDctOutputs, numOfMelFilters)
    
    num_samples = len(recordings_list)
    mfcc_features = np.empty((num_samples, numOfDctOutputs * 2), np.float32)
    labels = np.empty(num_samples)
    
    # Initialize MFCC instance
    mfccf32 = dsp.arm_mfcc_instance_f32()
    
    status = dsp.arm_mfcc_init_f32(
        mfccf32,
        FFTSize,
        numOfMelFilters,
        numOfDctOutputs,
        dctMatrixFilters,
        filtPos,
        filtLen,
        packedFilters,
        window,
    )
    
    # Process each recording
    for sample_idx, wav_path in enumerate(recordings_list):
        wav_file = os.path.basename(wav_path)
        file_specs = wav_file.split(".")[0]
        digit, person, recording = file_specs.split("_")
        
        # Read WAV file
        _, sample = wavfile.read(wav_path)
        sample = sample.astype(np.float32)[:2 * FFTSize]
        
        # Pad if necessary
        if len(sample) < 2 * FFTSize:
            sample = np.pad(sample, (0, 2 * FFTSize - len(sample)), "constant", constant_values=0)
        
        # Normalize audio
        sample = sample / max(abs(sample))
        
        # Split into two halves for feature extraction
        first_half = sample[:FFTSize]
        second_half = sample[FFTSize:2*FFTSize]
        
        # Extract MFCC features from both halves
        tmp = np.zeros(FFTSize + 2)
        first_half_mfcc = dsp.arm_mfcc_f32(mfccf32, first_half, tmp)
        second_half_mfcc = dsp.arm_mfcc_f32(mfccf32, second_half, tmp)
        
        # Concatenate features
        mfcc_feature = np.concatenate((first_half_mfcc, second_half_mfcc))
        mfcc_features[sample_idx] = mfcc_feature
        labels[sample_idx] = int(digit)
    
    return mfcc_features, labels

