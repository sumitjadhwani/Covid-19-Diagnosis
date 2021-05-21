import librosa
import numpy as np
import csv
import os
from app import APP_ROOT


def extractFeature(path, features):
    path = os.path.join(APP_ROOT, 'audio.wav')
    y, sr = librosa.load(path, mono=True)
    
    # Extracting features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    # Appends the features to 'featureList'
    featureList = []
    for i in [chroma_stft, spec_cent, spec_bw, rolloff, zcr, mfcc]:
        featureList.append(np.mean(i))
    
    featureList.append(features)
    return featureList