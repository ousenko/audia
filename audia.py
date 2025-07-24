#!/usr/bin/env python3
"""
Audia - Lightning Fast Audio Transcription and AI Processing

Main CLI entry point for the Lightning Whisper MLX transcription pipeline
with optional AI-powered transcript processing.

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
"""

import argparse
import sys
import logging
from pathlib import Path

# Import our modules
try:
    from pipeline import AudioTranscriptionPipeline, AIProcessor
except ImportError as e:
    print(f"❌ Error importing pipeline modules: {e}")
    sys.exit(1)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Audia - Ultra-fast Apple Silicon audio transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Use Cases:

1. TRANSCRIPTION ONLY:
  %(prog)s audio.m4a                          # Basic Russian transcription (large-v3 model)
  %(prog)s audio.m4a -m small                 # Fast transcription (small model, 23x real-time)
  %(prog)s audio.m4a -m medium                # Balanced transcription (medium model, 10x real-time)
  %(prog)s audio.m4a -l en                    # English transcription
  %(prog)s audio.m4a -f all                   # Output all formats (txt, json, srt)
  %(prog)s audio.m4a -f srt                   # Output subtitle format

2. TRANSCRIPTION + AI PROCESSING:
  %(prog)s audio.m4a -p meeting_notes         # Meeting notes generation
  %(prog)s audio.m4a -p podcast_summary       # Podcast summary generation
  %(prog)s audio.m4a -m medium -p meeting_notes  # Faster transcription + AI processing
        """
    )
    
    parser.add_argument("input", nargs='?', help="Input audio file path (M4A, MP3, WAV, etc.)")
    parser.add_argument("-o", "--output", help="Output path (without extension)")
    parser.add_argument("--output-dir", default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("-m", "--model", default="large-v3", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                       help="Whisper model to use (default: large-v3). All models support Russian language.")
    parser.add_argument("-l", "--language", default="ru", help="Language code (default: ru). Supports en, ru, es, etc.")
    parser.add_argument("-f", "--format", default="txt",
                       choices=["txt", "json", "srt", "formatted", "all"],
                       help="Output format (default: txt)")
    parser.add_argument("--batch-size", type=int, default=12,
                       help="Batch size for Lightning Whisper MLX processing (default: 12)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose logging")
    
    # AI Processing arguments
    parser.add_argument("-p", "--process", 
                       help="Enable AI processing with specified prompt (e.g., meeting_notes, podcast_summary)")
    parser.add_argument("--list-prompts", action="store_true",
                       help="List available AI processing prompts")
    
    args = parser.parse_args()
    
    # Handle list prompts command
    if args.list_prompts:
        if AIProcessor:
            try:
                processor = AIProcessor(require_api_key=False)
                prompts = processor.list_available_prompts()
                print("📝 Available AI processing prompts:")
                for prompt in prompts:
                    print(f"  • {prompt}")
                return
            except Exception as e:
                print(f"❌ Error loading prompts: {e}")
                return
        else:
            print("❌ AI processing not available. Install required dependencies.")
            return
    
    # Check if input file is required
    if not args.input and not args.list_prompts:
        parser.error("Input audio file is required unless using --list-prompts")
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine output path - save to specified output directory
    if args.output:
        output_path = Path(args.output)
    else:
        input_path = Path(args.input)
        # Create output directory if it doesn't exist
        outputs_dir = Path(args.output_dir)
        outputs_dir.mkdir(exist_ok=True)
        output_path = outputs_dir / input_path.stem
    
    try:
        # Create Lightning Whisper MLX pipeline
        print("🚀 Using Lightning Whisper MLX for ultra-fast Apple Silicon transcription")
        if args.process:
            print(f"🤖 AI processing enabled with prompt: {args.process}")
        
        pipeline = AudioTranscriptionPipeline(
            model_name=args.model,
            batch_size=args.batch_size,
            enable_ai_processing=bool(args.process)
        )
        
        # Process file
        result = pipeline.process_file(
            input_path=args.input,
            output_path=str(output_path),
            language=args.language,
            format_type=args.format,
            ai_prompt_type=args.process
        )
        
        # Print summary
        print(f"\n✅ Transcription completed successfully!")
        print(f"📁 Input: {args.input}")
        print(f"📄 Output: {output_path}.*")
        print(f"🌍 Language: {result.get('language', 'auto-detected')}")
        print(f"⏱️  Duration: {result.get('duration', 0):.1f} seconds")
        print(f"📝 Text length: {len(result.get('text', ''))} characters")
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
