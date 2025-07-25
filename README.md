# ⚡ audia

An ultra-fast audio transcription pipeline using Lightning Whisper MLX for maximum Apple Silicon performance. Achieves 6-23x real-time transcription speed while maintaining excellent quality.

## 🚀 Performance Highlights

- **Lightning small**: 23.3x real-time speed (6.3s for 147s audio) - Maximum speed
- **Lightning medium**: 10.7x real-time speed (13.7s for 147s audio) - **RECOMMENDED**
- **Lightning large-v3**: 5.9x real-time speed (24.8s for 147s audio) - **PREMIUM**
- **Russian language**: Excellent transcription quality with proper formatting
- **Apple Silicon**: Optimized exclusively for M1/M2/M3 chips
- **Multiple Output Formats**: TXT (default), JSON, SRT, formatted transcripts

## 📊 Speed Comparison

| Method | Model | Speed | Use Case |
|--------|-------|-------|----------|
| Lightning MLX | small | **23x real-time** | Quick drafts, maximum speed |
| Lightning MLX | medium | **10x real-time** | Balanced speed/quality |
| Lightning MLX | large-v3-turbo | **~15x real-time** | Fast + high quality |
| Lightning MLX | large-v3 | **6x real-time** | **Default** - best quality |

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

## 🚀 Quick Start

### Option 1: Convenience Wrapper (Recommended)
Use the `audia` wrapper script that automatically handles virtual environment:

```bash
# Use Case 1: Transcription Only
./audia audio.m4a                    # Basic Russian transcription
./audia audio.m4a -m small           # Fast transcription (23x real-time)
./audia audio.m4a -l en              # English transcription

# Use Case 2: Transcription + AI Processing
./audia audio.m4a -p meeting_notes   # Meeting notes generation
./audia --list-prompts               # List available AI prompts
```

### Option 2: Direct Python Call
If you prefer to manage virtual environment manually:

```bash
# Activate virtual environment first
source venv/bin/activate

# Use Case 1: Transcription Only
python audia.py audio.m4a

# Use Case 2: Transcription + AI Processing
python audia.py audio.m4a -p meeting_notes
```

## 💻 Usage

### Use Case 1: Transcription Only

```bash
# Basic Russian transcription (large-v3 model, 6x real-time)
./audia audio.m4a

# Fast transcription (small model - 23x real-time)
./audia audio.m4a -m small

# Balanced quality/speed (medium model - 10x real-time)
./audia audio.m4a -m medium

# Fast + high quality (large-v3-turbo model - ~15x real-time)
./audia audio.m4a -m large-v3-turbo

# English language transcription
./audia audio.m4a -l en

# Save to specific file
./audia audio.m4a -o transcript.txt

# Save to custom path
./audia audio.m4a -o /path/to/my_transcript.txt

# Output all formats (txt, json, srt)
./audia audio.m4a -f all
```

### Use Case 2: Transcription + AI Processing

```bash
# Meeting notes generation
./audia audio.m4a -p meeting_notes

# Podcast summary generation
./audia audio.m4a -p podcast_summary

# Faster transcription + AI processing
./audia audio.m4a -m medium -p meeting_notes

# List available AI prompts
./audia --list-prompts
```


## ⚙️ Command Line Options

| Parameter | Description | Example |
|-----------|-------------|----------|
| `-o, --output` | Output file path | `-o transcript.txt` |
| `-m, --model` | Whisper model | `-m large-v3-turbo` |
| `-l, --language` | Audio language | `-l en` (default: ru) |
| `-f, --format` | Output format | `-f all` (default: txt) |
| `-p, --process` | AI processing prompt | `-p meeting_notes` |
| `--list-prompts` | Show available prompts | `--list-prompts` |
| `--output-dir` | Output directory | `--output-dir results` |
| `--batch-size` | Processing batch size | `--batch-size 16` |

## 🔧 Environment Setup

### Configure API Keys

```bash
# Copy template and edit with your keys
cp .env.example .env
```

```env
OPENAI_API_URL=https://api.openai.com/v1/chat/completions
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4o-mini
```

## 📁 Output Files

By default, only a plain text transcript is created. Use `-f all` for multiple formats:


### File Format Details

- **Plain Text** (`outputs/audio.txt`): Clean, formatted transcript without timestamps
- **JSON** (`outputs/audio.json`): Complete transcription data with timestamps and metadata
- **SRT** (`outputs/audio.srt`): Standard subtitle format for video players
- **Formatted** (`outputs/audio.formatted.txt`): Human-readable transcript with timestamps
- **AI Processed** (`outputs/audio.meeting_notes.md`): AI-generated structured notes

### Custom Output Directory

```bash
# Use different output directory
./audia audio.m4a --output-dir my_results

# Specify exact output path
./audia audio.m4a -o /path/to/specific/output
```

## 🎯 Features

- **Ultra-fast transcription** using Lightning Whisper MLX optimized for Apple Silicon
- **AI-powered transcript processing** with customizable prompts for meeting notes, podcast summaries, and more
- **Multiple output formats**: Plain text, JSON, SRT subtitles, formatted transcripts, AI-processed summaries
- **High accuracy** with support for 99+ languages including excellent Russian support
- **Batch processing** with configurable batch sizes for optimal performance
- **Smart audio preprocessing** with automatic format conversion and noise handling
- **Memory efficient** processing of large audio files
- **Flexible AI integration** using OpenAI-compatible APIs for transcript analysis
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

## 🤖 AI Processing Setup

Edit `config.yaml` to customize AI processing:

```yaml
ai_processing:
  # Default API URL (overridden by OPENAI_API_URL environment variable)
  api_url: "https://api.openai.com/v1/chat/completions"
  model: "gpt-4o-mini"
  temperature: 0.3
  max_tokens: 4000
  timeout: 60
```

### Custom Prompts

Create custom AI processing prompts in the `prompts/` directory:

```
prompts/
├── meeting_notes.txt          # Meeting notes format
├── podcast_summary.txt        # Podcast summary format
└── your_custom_prompt.txt     # Your custom format
```

**Usage:**
```bash
# List all available prompts
./audia --list-prompts

# Use your custom prompt
./audia audio.m4a -p your_custom_prompt
```

## 🚨 Requirements

- **Hardware**: Apple Silicon Mac (M1/M2/M3)
- **OS**: macOS with Metal support
- **Python**: 3.8+
- **Dependencies**: MLX, Lightning Whisper MLX, FFmpeg


## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### Apache License 2.0

Copyright 2025 Andrew Ousenko

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
