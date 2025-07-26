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
import os
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import dotenv_values
    env_vars = dotenv_values('.env')
except ImportError:
    # dotenv is optional
    env_vars = {}

# Import our modules
try:
    from pipeline import AudioTranscriptionPipeline, AIProcessor
except ImportError as e:
    print(f"❌ Error importing pipeline modules: {e}")
    sys.exit(1)


def process_existing_transcript(transcript_path: Path, prompt_type: str, output_path: str = None, output_dir: str = "outputs", verify_ssl: bool = True):
    """
    Process an existing transcript file with AI prompt.
    
    Args:
        transcript_path: Path to the transcript file
        prompt_type: AI prompt type to use
        output_path: Optional custom output path
        output_dir: Output directory (default: outputs)
        verify_ssl: SSL verification setting
    
    Returns:
        Path to the processed output file
    """
    print(f"🤖 Processing transcript with AI prompt: {prompt_type}")
    print(f"📄 Input: {transcript_path}")
    
    # Read the transcript file
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_text = f.read().strip()
        
        if not transcript_text:
            raise ValueError("Transcript file is empty")
            
        print(f"📝 Transcript length: {len(transcript_text)} characters")
        
    except Exception as e:
        raise Exception(f"Failed to read transcript file: {e}")
    
    # Initialize AI processor
    try:
        if not AIProcessor:
            raise Exception("AI processing not available. Install required dependencies.")
            
        processor = AIProcessor()
        
    except Exception as e:
        raise Exception(f"Failed to initialize AI processor: {e}")
    
    # Determine output path
    if output_path:
        final_output_path = Path(output_path)
        # Remove extension if provided and add .txt
        if final_output_path.suffix:
            final_output_path = final_output_path.with_suffix('.txt')
        else:
            final_output_path = final_output_path.with_suffix('.txt')
    else:
        # Create output filename based on input and prompt
        base_name = transcript_path.stem
        final_output_path = Path(output_dir) / f"{base_name}_{prompt_type}.txt"
    
    # Create output directory if it doesn't exist
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process the transcript with AI
    try:
        print(f"🧠 Processing with AI...")
        processed_text = processor.process_transcript(transcript_text, prompt_type)
        
        if not processed_text:
            raise Exception("AI processing returned empty result")
            
        # Save the processed result
        with open(final_output_path, 'w', encoding='utf-8') as f:
            f.write(processed_text)
            
        print(f"💾 Saved processed result to: {final_output_path}")
        return str(final_output_path)
        
    except Exception as e:
        raise Exception(f"AI processing failed: {e}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Audia - Ultra-fast Apple Silicon audio transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Use Cases:

1. TRANSCRIPTION ONLY:
  %(prog)s audio.m4a -o transcript.txt        # Basic Russian transcription (large-v3-turbo model)
  %(prog)s audio.m4a -o transcript.txt -m small  # Fast transcription (small model, 23x real-time)
  %(prog)s audio.m4a -o transcript.txt -m medium # Balanced transcription (medium model, 10x real-time)
  %(prog)s audio.m4a -o transcript.txt -l en  # English transcription
  %(prog)s audio.m4a -o /path/to/result.txt   # Save to custom path
  %(prog)s audio.m4a -o transcript.txt -f all # Output all formats (txt, json, srt)

2. TRANSCRIPTION + AI PROCESSING:
  %(prog)s audio.m4a -o transcript.txt -p meeting_notes    # Meeting notes generation
  %(prog)s audio.m4a -o transcript.txt -p podcast_summary # Podcast summary generation
  %(prog)s audio.m4a -o transcript.txt -m medium -p meeting_notes # Faster transcription + AI processing

3. TRANSCRIPT POST-PROCESSING:
  %(prog)s --process-transcript transcript.txt -p meeting_notes  # Process existing transcript
  %(prog)s --process-transcript transcript.txt -p psy -o analysis.txt  # Custom output file
  %(prog)s --process-transcript outputs/audio.txt -p podcast_summary  # Process previous result

        """
    )
    
    parser.add_argument("input", nargs='?', help="Input audio file path (M4A, MP3, WAV, etc.)")
    parser.add_argument("-o", "--output", required=True, help="Output file path (e.g., /path/to/transcript.txt)")
    parser.add_argument("--output-dir", default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("-m", "--model", default="large-v3-turbo", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "large-v3-turbo"],
                       help="Whisper model to use (default: large-v3-turbo). All models support Russian language.")
    parser.add_argument("-l", "--language", default="ru", help="Language code (default: ru). Supports en, ru, es, etc.")
    parser.add_argument("-f", "--format", default="txt",
                       choices=["txt", "json", "srt", "formatted", "all"],
                       help="Output format (default: txt)")
    parser.add_argument("--batch-size", type=int, default=12,
                       help="Batch size for Lightning Whisper MLX processing (default: 12)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--no-ssl-verify", action="store_true",
                       help="Disable SSL verification for model downloads (for corporate environments). Can also set AUDIA_NO_SSL_VERIFY=true in .env file")
    
    # AI Processing arguments
    parser.add_argument("-p", "--process", 
                       help="Enable AI processing with specified prompt (e.g., meeting_notes, podcast_summary)")
    parser.add_argument("--process-transcript", metavar="TRANSCRIPT_FILE",
                       help="Process an existing transcript file with AI prompt (requires -p/--process)")
    parser.add_argument("--list-prompts", action="store_true",
                       help="List available AI processing prompts")
    
    args = parser.parse_args()
    
    # Check for SSL verification settings (.env file overrides command line)
    no_ssl_verify = args.no_ssl_verify
    if env_vars.get('AUDIA_NO_SSL_VERIFY', '').lower() in ('true', '1', 'yes'):
        no_ssl_verify = True
    
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
    
    # Handle transcript processing mode
    if args.process_transcript:
        if not args.process:
            parser.error("--process-transcript requires -p/--process to specify the prompt")
        
        # Check if transcript file exists
        transcript_path = Path(args.process_transcript)
        if not transcript_path.exists():
            print(f"❌ Error: Transcript file not found: {transcript_path}")
            return
        
        # Process the transcript
        try:
            result = process_existing_transcript(
                transcript_path=transcript_path,
                prompt_type=args.process,
                output_path=args.output,
                output_dir=args.output_dir,
                verify_ssl=not no_ssl_verify
            )
            if result:
                print(f"✅ Transcript processing completed successfully!")
                print(f"📄 Output: {result}")
            return
        except Exception as e:
            print(f"❌ Error processing transcript: {e}")
            return
    
    # Check if input file is required for audio transcription
    if not args.input and not args.list_prompts:
        parser.error("Input audio file is required unless using --list-prompts or --process-transcript")
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove extension if provided, we'll add it based on format
        if output_path.suffix:
            output_path = output_path.with_suffix('')
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
            enable_ai_processing=bool(args.process),
            verify_ssl=not no_ssl_verify
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
