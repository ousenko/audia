# ⚡ Lightning Whisper MLX Audio Transcription Pipeline

An ultra-fast audio transcription pipeline using Lightning Whisper MLX for maximum Apple Silicon performance. Achieves 6-23x real-time transcription speed while maintaining excellent quality.

## 🚀 Performance Highlights

- **Lightning small**: 23.3x real-time speed (6.3s for 147s audio) - Maximum speed
- **Lightning medium**: 10.7x real-time speed (13.7s for 147s audio) - **RECOMMENDED**
- **Lightning large-v3**: 5.9x real-time speed (24.8s for 147s audio) - **PREMIUM**
- **Russian language**: Excellent transcription quality with proper formatting
- **Apple Silicon**: Optimized exclusively for M1/M2/M3 chips
- **Multiple Output Formats**: TXT, JSON, SRT, formatted transcripts

## 📊 Speed Comparison

| Method | Model | Speed | Use Case |
|--------|-------|-------|----------|
| Lightning MLX | small | **23x real-time** | Quick drafts, maximum speed |
| Lightning MLX | medium | **10x real-time** | **Recommended** - balanced speed/quality |
| Standard MLX | large-v3 | 0.6x real-time | Maximum accuracy |

## 🛠️ Installation

### Prerequisites
- **Apple Silicon Mac** (M1/M2/M3) - Required for Lightning Whisper MLX
- **Python 3.8+**
- **FFmpeg** for audio processing

### Install Dependencies

```bash
# Install FFmpeg (required for audio processing)
brew install ffmpeg

# Install Lightning Whisper MLX and dependencies
pip install lightning-whisper-mlx
pip install -r requirements.txt
```

## 💻 Usage

### Ultra-Fast Transcription (Lightning MLX)

```bash
# Maximum speed - 23x real-time

# Fast transcription (small model - 23x real-time)
python audio_transcription_pipeline.py audio.m4a -m small

# Premium quality (large-v3 model - 6x real-time)
python audio_transcription_pipeline.py audio.m4a -m large-v3

# Russian language transcription
python audio_transcription_pipeline.py audio.m4a -l ru

# Custom output format
python audio_transcription_pipeline.py audio.m4a -f srt

# Optimize batch size for speed
python audio_transcription_pipeline.py audio.m4a --batch-size 16
```

### Output Format Options

```bash
# All formats (default)
python audio_transcription_pipeline.py audio.m4a

# Specific format
python audio_transcription_pipeline.py audio.m4a -f formatted

# Multiple formats
python audio_transcription_pipeline.py audio.m4a -f json
```

## 📁 Output Files

The pipeline generates multiple output formats:

- `audio.txt` - Plain text transcript
- `audio.json` - Detailed JSON with timestamps and metadata
- `audio.srt` - SRT subtitle format
- `audio.formatted.txt` - Clean, formatted transcript with timestamps

## 🎯 Features

- **Lightning Whisper MLX Integration**: Ultra-fast transcription with 10-23x real-time speed
- **Apple Silicon Optimization**: Native MLX framework acceleration
- **Audio Normalization**: FFmpeg-based preprocessing matching vibe quality
- **Multiple Model Support**: tiny, base, small, medium, large, large-v3
- **Language Detection**: Automatic language detection or manual specification
- **Batch Processing**: Configurable batch sizes for Lightning MLX
- **Error Handling**: Robust error handling and logging
- **CLI Interface**: Easy-to-use command-line interface

## 🔧 Technical Details

### Audio Processing
- Automatic normalization using FFmpeg loudnorm filter
- 16kHz, mono, 16-bit PCM conversion
- Compatible with M4A, MP3, WAV, and other formats

### Model Support
- **Lightning MLX Models**: Optimized for maximum speed
- **Standard MLX Models**: Full feature set with high accuracy
- **Quantization**: 4-bit and 8-bit quantization support (when available)

### Apple Silicon Optimization
- MLX framework for Metal GPU acceleration
- Neural Engine utilization (model dependent)
- Memory-efficient processing
- FP16 inference for speed

## 📈 Performance Benchmarks

Tested on Apple Silicon M3 with 147-second Russian audio:

| Configuration | Time | Speed | Quality |
|---------------|------|-------|---------|
| Lightning + small | 6.3s | 23.3x | Good |
| Lightning + medium | 13.7s | 10.7x | Better |
| Standard + small | 64s | 2.3x | Good |
| Standard + medium | 169s | 0.87x | Better |
| Standard + large-v3 | 240s+ | 0.6x | Best |

## 🌍 Language Support

Optimized for Russian language but supports all Whisper languages:
- Russian (ru) - Primary focus
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- And 90+ other languages

## 🚨 Requirements

- **Hardware**: Apple Silicon Mac (M1/M2/M3)
- **OS**: macOS with Metal support
- **Python**: 3.8+
- **Dependencies**: MLX, Lightning Whisper MLX, FFmpeg

## 🤝 Contributing

This pipeline is designed to match and exceed vibe application performance on Apple Silicon. Contributions welcome for:

- Additional model optimizations
- Core ML integration
- Batch processing improvements
- New output formats

## 📄 License

[Add your license here]

---

**Achieving vibe-level transcription performance with 10-23x real-time speed on Apple Silicon** 🚀
