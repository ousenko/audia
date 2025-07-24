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

## 🚀 Quick Start

```bash
# Basic transcription (medium model, 10x real-time)
python audia.py audio.m4a

# Maximum speed (small model, 23x real-time)
python audia.py audio.m4a -m small

# Premium quality (large-v3 model, 6x real-time)
python audia.py audio.m4a -m large-v3

# With AI processing
python audia.py audio.m4a --ai-process --ai-prompt meeting_notes

# List available AI prompts
python audia.py --list-prompts
```

## 💻 Usage

### Ultra-Fast Transcription (Lightning MLX)

```bash
# Maximum speed - 23x real-time

# Fast transcription (small model - 23x real-time)
python audia.py audio.m4a -m small

# Premium quality (large-v3 model - 6x real-time)
python audia.py audio.m4a -m large-v3

# Russian language transcription
python audia.py audio.m4a -l ru

# Custom output format
python audia.py audio.m4a -f srt

# Optimize batch size for speed
python audia.py audio.m4a --batch-size 16

# AI-powered transcript processing
python audia.py audio.m4a --ai-process --ai-prompt meeting_notes
python audia.py audio.m4a --ai-process --ai-prompt podcast_summary
```

### AI Processing Features

```bash
# List available AI prompts
python audia.py --list-prompts

# Basic transcription with AI processing
python audia.py audio.m4a --ai-process

# Use specific AI prompt
python audia.py audio.m4a --ai-process --ai-prompt podcast_summary

# Combine with transcription options
python audia.py audio.m4a -m large-v3 --ai-process --ai-prompt meeting_notes
```

## 🎯 Common Use Cases

### Meeting Processing
1. Record your meeting in any audio format
2. Run: `python audia.py meeting.m4a --ai-process --ai-prompt meeting_notes`
3. Get structured notes with participants, topics, decisions, and action items

### Podcast/Interview Processing
1. Run: `python audia.py interview.mp3 --ai-process --ai-prompt podcast_summary`
2. Get summary with key insights and quotes

### Lecture/Presentation Processing
1. Run: `python audia.py lecture.wav --ai-process --ai-prompt meeting_notes`
2. Get structured notes and key points

## ⚙️ Command Line Options

| Parameter | Description | Example |
|-----------|-------------|----------|
| `--model` | Whisper model | `--model large-v3` |
| `--language` | Audio language | `--language ru` |
| `--format` | Output format | `--format json` |
| `--ai-process` | Enable AI processing | `--ai-process` |
| `--ai-prompt` | AI prompt type | `--ai-prompt meeting_notes` |
| `--list-prompts` | Show available prompts | `--list-prompts` |
| `--output-dir` | Output directory | `--output-dir results` |
| `--batch-size` | Processing batch size | `--batch-size 16` |

## 🔧 Environment Setup

### Configure API Keys

```bash
# Copy template and edit with your keys
cp .env.example .env
```

**For OpenAI:**
```env
OPENAI_API_URL=https://api.openai.com/v1/chat/completions
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4o-mini
```

**For X.AI (Grok):**
```env
OPENAI_API_URL=https://api.x.ai/v1/chat/completions
OPENAI_API_KEY=xai-your-xai-key-here
OPENAI_MODEL=grok-beta
```

**For other OpenAI-compatible APIs:**
```env
OPENAI_API_URL=https://your-api.com/v1/chat/completions
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=your-model-name
```

## 📁 Output Files

All output files are saved to the `outputs/` directory by default:

```
audio.m4a                           # Input file
outputs/                            # Output directory
├── audio.txt                       # Plain text transcript
├── audio.json                      # JSON with timestamps
├── audio.srt                       # Subtitle format
├── audio.formatted.txt             # Formatted transcript
└── audio.meeting_notes.md          # AI-processed notes
```

### File Format Details

- **Plain Text** (`outputs/audio.txt`): Clean, formatted transcript without timestamps
- **JSON** (`outputs/audio.json`): Complete transcription data with timestamps and metadata
- **SRT** (`outputs/audio.srt`): Standard subtitle format for video players
- **Formatted** (`outputs/audio.formatted.txt`): Human-readable transcript with timestamps
- **AI Processed** (`outputs/audio.meeting_notes.md`): AI-generated structured notes

### Custom Output Directory

```bash
# Use different output directory
python audia.py audio.m4a --output-dir my_results

# Specify exact output path
python audia.py audio.m4a -o /path/to/specific/output
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

To use AI-powered transcript processing, you need to configure an OpenAI-compatible API:

### 1. Set Environment Variables

```bash
# Set your OpenAI API key (required)
export OPENAI_API_KEY="your-api-key-here"

# Set custom API URL (optional, defaults to OpenAI)
export OPENAI_API_URL="https://api.openai.com/v1/chat/completions"
```

### 2. Configure API Settings (Optional)

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

## 📚 Practical Examples

### Meeting Recording Processing
```bash
# Transcription + structured notes
python audia.py meeting_2024_07_24.m4a --ai-process --ai-prompt meeting_notes

# Result: meeting_2024_07_24.meeting_notes.md
# Contains: participants, key topics, decisions, action items
```

### Podcast Processing
```bash
# Transcription + summary
python audia.py podcast_episode_42.mp3 --ai-process --ai-prompt podcast_summary

# Result: podcast_episode_42.podcast_summary.md
# Contains: summary, key insights, notable quotes
```

### Batch Processing
```bash
# Process multiple files
for file in *.m4a; do
    python audia.py "$file" --ai-process --ai-prompt meeting_notes
done
```

## 🔍 Troubleshooting

### "API key not found" Error
- Check your `.env` file exists and contains `OPENAI_API_KEY`
- Ensure the key is valid and has sufficient credits
- Verify the API URL matches your provider

### "Model not found" Error
- Check the model name in your `.env` file
- For X.AI use `grok-beta` or `grok-2`
- For OpenAI use `gpt-4o-mini` or `gpt-4`

### Slow Processing
- Use `medium` model instead of `large-v3` for faster processing
- Specify language with `--language ru` for better performance
- Increase batch size with `--batch-size 16`

### Poor Transcription Quality
- Use `large-v3` model for maximum accuracy
- Ensure good audio quality (clear speech, minimal background noise)
- Specify the correct language code

## 💡 Tips & Best Practices

### Performance Optimization
- **small model**: Maximum speed (23x real-time) for quick drafts
- **medium model**: Balanced speed/quality (10x real-time) - recommended
- **large-v3 model**: Maximum accuracy (6x real-time) for important content

### AI Processing Quality
- Long transcripts are automatically chunked for processing
- Results are synthesized into a unified report
- Create custom prompts in the `prompts/` directory

### Security
- API keys are stored in `.env` file (excluded from git)
- Never commit your `.env` file to version control
- Use `.env.example` as a template

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
python audia.py --list-prompts

# Use your custom prompt
python audia.py audio.m4a --ai-process --ai-prompt your_custom_prompt
```

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

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### Apache License 2.0

Copyright 2025 Lightning Whisper MLX Audio Transcription Pipeline

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

**Achieving vibe-level transcription performance with 10-23x real-time speed on Apple Silicon** 🚀
