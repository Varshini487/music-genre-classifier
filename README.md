# 🎵 Music Genre Classifier

Classify music genre (Rock, Pop, Classical, Hip-Hop, Jazz, Blues, Country, Reggae) from raw audio waveforms using Convolutional Neural Networks.

## Challenge
Raw audio is 44,100+ samples per second. A 3-minute song = 7.9M samples. CNNs need structured input.

## Solution: Spectrogram
- Convert time-domain audio → frequency-domain spectrogram
- Apply Short-Time Fourier Transform (STFT)
- Spectrogram = 2D image (frequency × time)
- CNNs love images → classify genres

## Architecture
```
Raw Audio (44.1kHz)
    ↓
STFT → Mel-Spectrogram (128 × 646)
    ↓
ResNet50 (pre-trained on ImageNet)
    ↓
Fine-tune last 2 layers on GTZAN dataset
    ↓
8-class genre classifier
```

## Dataset
- GTZAN Genre Collection: 1,000 30-second clips (10 genres)
- Mel-Spectrogram: 128 frequency bins, 646 time steps

## Tech Stack
- **librosa** – audio loading + mel-spectrogram
- **TensorFlow/Keras** – ResNet50 fine-tuning
- **Matplotlib** – spectrogram visualization
- **Streamlit** – upload interface

## Performance
- Accuracy: 88% on test set
- Confusion matrix: Pop/Rock often confused (similar instruments)
- Classical/Jazz well-separated (distinct frequency distributions)

## Use Cases
- Music streaming platform recommendations
- Automated music library tagging
- DJ mix analysis
- Music education tools (identify genre by ear)
