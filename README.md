# ⚡ audia

An ultra-fast audio and video transcription pipeline using Lightning Whisper MLX for maximum Apple Silicon performance. Achieves 6-23x real-time transcription speed while maintaining excellent quality. Supports automatic audio extraction from video files.

## 🚀 Performance Highlights

- **Lightning small**: 23.3x real-time speed (6.3s for 147s audio) - Maximum speed
- **Lightning large-v3-turbo**: ~15x real-time speed - **DEFAULT** - fast + high quality
- **Lightning medium**: 10.7x real-time speed (13.7s for 147s audio) - balanced
- **Lightning large-v3**: 5.9x real-time speed (24.8s for 147s audio) - best quality
- **Russian language**: Excellent transcription quality with proper formatting
- **Apple Silicon**: Optimized exclusively for M1/M2/M3 chips
- **Multiple Output Formats**: Markdown (default), JSON, SRT, formatted transcripts

## 📊 Speed Comparison

| Method | Model | Speed | Use Case |
|--------|-------|-------|----------|
| Lightning MLX | tiny | **~30x real-time** | Ultra-fast, basic quality |
| Lightning MLX | base | **~25x real-time** | Fast, improved quality |
| Lightning MLX | small | **23x real-time** | Quick drafts, good quality |
| Lightning MLX | medium | **10x real-time** | Balanced speed/quality |
| Lightning MLX | large | **~8x real-time** | High quality |
| Lightning MLX | large-v2 | **~7x real-time** | Enhanced quality |
| Lightning MLX | large-v3-turbo | **~15x real-time** | **Default** - fast + high quality |
| Lightning MLX | large-v3 | **6x real-time** | Best quality |

## 🛠️ Installation

### Prerequisites
- **Apple Silicon Mac** (M1/M2/M3) - Required for Lightning Whisper MLX
- **Python 3.8+**
- **FFmpeg** for audio processing

### Recommended Installation (with virtual environment)

```bash
# Install FFmpeg (required for audio processing)
brew install ffmpeg

# Create and activate virtual environment (RECOMMENDED)
python -m venv venv
source venv/bin/activate

# Install Lightning Whisper MLX and dependencies
pip install lightning-whisper-mlx
pip install -r requirements.txt
```

## 💻 Usage

### Use Case 1: Transcription Only

```bash
# Basic Russian transcription (large-v3-turbo model, ~15x real-time)
./audia audio.m4a -o transcript.md

# Video transcription (automatic audio extraction)
./audia video.mp4 -o transcript.md
./audia meeting.mov -o transcript.md
./audia presentation.avi -o transcript.md

# Fast transcription (small model - 23x real-time)
./audia audio.m4a -o transcript.md -m small
./audia video.mp4 -o transcript.md -m small

# Balanced quality/speed (medium model - 10x real-time)
./audia audio.m4a -o transcript.md -m medium

# Best quality (large-v3 model - 6x real-time)
./audia audio.m4a -o transcript.md -m large-v3

# English language transcription
./audia audio.m4a -o transcript.md -l en
./audia video.mp4 -o transcript.md -l en

# Save to custom path
./audia audio.m4a -o /path/to/my_transcript.md

# Output all formats (md, json, srt)
./audia audio.m4a -o transcript.md -f all
```

### Use Case 2: Transcription + AI Processing

```bash
# Meeting notes generation
./audia audio.m4a -o transcript.md -p meeting_notes
./audia meeting.mp4 -o transcript.md -p meeting_notes

# Podcast summary generation
./audia audio.m4a -o transcript.md -p podcast_summary
./audia podcast.mov -o transcript.md -p podcast_summary

# Faster transcription + AI processing
./audia audio.m4a -o transcript.md -m medium -p meeting_notes
./audia video.mp4 -o transcript.md -m medium -p meeting_notes

# List available AI prompts
./audia --list-prompts
```

### Use Case 3: Transcript Post-Processing

```bash
# Process existing transcript with AI prompt
./audia --process-transcript transcript.md -p meeting_notes

# Process with custom output file
./audia --process-transcript transcript.md -p psy -o analysis.md

# Process previous transcription result
./audia --process-transcript outputs/audio.md -p podcast_summary

# Process transcript from different directory
./audia --process-transcript /path/to/transcript.md -p meeting_notes
```

## ⚙️ Command Line Options

| Parameter | Description | Default | Example |
|-----------|-------------|---------|----------|
| `input` | Input audio/video file | - | `audio.m4a` |
| `-o, --output` | Output file path (required) | - | `-o transcript.md` |
| `-m, --model` | Whisper model | `large-v3-turbo` | `-m small` |
| `-l, --language` | Audio language | `ru` | `-l en` |
| `-f, --format` | Output format | `md` | `-f all` |
| `-p, --process` | AI processing prompt | - | `-p meeting_notes` |
| `--process-transcript` | Process existing transcript | - | `--process-transcript file.md` |
| `--list-prompts` | Show available prompts | - | `--list-prompts` |
| `--output-dir` | Output directory | `outputs` | `--output-dir results` |
| `--batch-size` | Processing batch size | `12` | `--batch-size 16` |
| `-v, --verbose` | Enable verbose logging | `false` | `-v` |
| `--no-ssl-verify` | Disable SSL verification | `false` | `--no-ssl-verify` |

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

- **Markdown** (`outputs/audio.md`): Clean, structured transcript in Markdown format (default)
- **JSON** (`outputs/audio.json`): Complete transcription data with timestamps and metadata
- **SRT** (`outputs/audio.srt`): Standard subtitle format for video players
- **Formatted** (`outputs/audio.formatted.md`): Human-readable transcript with timestamps
- **AI Processed** (`outputs/audio.meeting_notes.md`): AI-generated structured notes

### Custom Output Directory

```bash
# Use different output directory
./audia audio.m4a -o transcript.md --output-dir my_results

# Specify exact output path
./audia audio.m4a -o /path/to/specific/output
```

## 🎯 Features

- **Ultra-fast transcription** using Lightning Whisper MLX optimized for Apple Silicon
- **Video file support** with automatic audio extraction from MP4, MOV, AVI, MKV, WebM, and more
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

## 🏢 Corporate Environment Setup

For corporate environments with self-signed certificates or proxy issues:

### SSL Certificate Issues
If you encounter SSL errors during model downloads:

```bash
# Option 1: CLI flag
./audia audio.m4a -o transcript.md --no-ssl-verify

# Option 2: Environment variable (persistent)
echo "AUDIA_NO_SSL_VERIFY=true" >> .env
./audia audio.m4a -o transcript.md
```

### Configuration in .env file
```env
# SSL settings for corporate environments
AUDIA_NO_SSL_VERIFY=true
```

**Important**: 
- Priority: `.env` file overrides CLI flag
- Scope: Affects Lightning Whisper MLX model downloads via huggingface_hub
- Security: SSL verification is disabled only for model downloads, not for API requests

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
./audia audio.m4a -o transcript.md -p your_custom_prompt
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
