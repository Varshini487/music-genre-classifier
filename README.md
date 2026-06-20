# 🎵 Music Genre Classifier

Classify music genre (Rock, Pop, Classical, Hip-Hop, Jazz, etc.) from audio using CNN on Mel-Spectrograms.

## 🧠 How It Works
1. Convert 30-second audio to Mel-Spectrogram (frequency × time image)
2. ResNet50 CNN processes the spectrogram as an image
3. Output: genre probabilities

## 🎵 Supported Genres
Rock, Pop, Classical, Hip-Hop, Jazz, Metal, Ambient, Electronic

## 📊 Performance
- Accuracy: 88%
- Model: ResNet50 (transfer learning from ImageNet)
- Dataset: GTZAN (1,000 songs across 10 genres)

## 🛠️ Tech Stack
- **librosa** – STFT, Mel-Spectrogram extraction
- **TensorFlow/Keras** – ResNet50 model
- **Matplotlib** – visualization
- **Streamlit** – web interface

## 🚀 Getting Started
```bash
git clone https://github.com/Varshini487/music-genre-classifier
cd music-genre-classifier
pip install -r requirements.txt
streamlit run app.py
```

## 💡 Use Cases
- Music streaming (auto-categorization)
- DJ/radio automation
- Music recommendation systems
- Podcast episode type detection
