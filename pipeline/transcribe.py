#!/usr/bin/env python3
"""
Audio Transcription Processing Pipeline
Reproduces the functionality of the vibe project for audio transcription.

This pipeline processes M4A audio files and produces clean, formatted transcripts
using OpenAI's Whisper model with quality similar to the vibe application.

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

import os
import sys
import logging
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import tempfile
import subprocess
import math
import ssl
import requests

# Lightning Whisper MLX - ultra-fast Apple Silicon optimization
try:
    from lightning_whisper_mlx import LightningWhisperMLX
except ImportError as e:
    print(f"❌ Error: Lightning Whisper MLX not found: {e}")
    print("")
    print("This pipeline requires Lightning Whisper MLX for maximum performance.")
    print("Please install it with:")
    print("  pip install lightning-whisper-mlx")
    print("")
    print("Note: Lightning Whisper MLX only works on Apple Silicon Macs (M1/M2/M3).")
    print("If you're not on Apple Silicon, this pipeline won't work.")
    sys.exit(1)

import torch
import torchaudio
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence

# AI Processing
try:
    from .process import AIProcessor
except ImportError:
    AIProcessor = None


class AudioTranscriptionPipeline:
    """Audio transcription pipeline using Lightning Whisper MLX for maximum Apple Silicon performance."""
    
    def __init__(self, model_name: str = "medium", batch_size: int = 12, quantization: Optional[str] = None, enable_ai_processing: bool = False, verify_ssl: bool = True):
        """
        Initialize the Lightning Whisper MLX transcription pipeline.
        
        Args:
            model_name: Whisper model to use (tiny, base, small, medium, large, large-v2, large-v3)
            batch_size: Batch size for Lightning processing (default: 12)
            quantization: Quantization level (None, '4bit', '8bit') - None for best compatibility
            enable_ai_processing: Enable AI-powered transcript processing
            verify_ssl: Enable SSL verification for model downloads (default: True)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.quantization = quantization
        self.verify_ssl = verify_ssl
        self.model = None
        self.sample_rate = 16000  # Whisper's expected sample rate
        
        # Configure logging first
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # AI Processing
        self.enable_ai_processing = enable_ai_processing
        self.ai_processor = None
        if enable_ai_processing and AIProcessor:
            try:
                self.ai_processor = AIProcessor()
                self.logger.info("AI processor initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize AI processor: {e}")
                self.enable_ai_processing = False
        
    def load_model(self) -> None:
        """Load the Lightning Whisper MLX model."""
        if self.model is None:
            self.logger.info(f"Loading Lightning Whisper MLX model '{self.model_name}'")
            self.logger.info("Using Lightning Whisper MLX for ultra-fast Apple Silicon performance")
            
            # Configure SSL for model downloads
            if not self.verify_ssl:
                self.logger.warning("SSL verification disabled for model downloads - this is insecure!")
                
                try:
                    from huggingface_hub import configure_http_backend
                    import urllib3
                    import os
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    
                    # Set environment variables to bypass proxy and SSL issues
                    os.environ['PYTHONHTTPSVERIFY'] = '0'
                    os.environ['CURL_CA_BUNDLE'] = ''
                    os.environ['REQUESTS_CA_BUNDLE'] = ''
                    os.environ['SSL_CERT_FILE'] = ''
                    
                    # Add huggingface.co to NO_PROXY to bypass proxy issues
                    current_no_proxy = os.environ.get('NO_PROXY', '')
                    if 'huggingface.co' not in current_no_proxy:
                        if current_no_proxy:
                            os.environ['NO_PROXY'] = f"{current_no_proxy},huggingface.co,*.huggingface.co"
                        else:
                            os.environ['NO_PROXY'] = "huggingface.co,*.huggingface.co"
                        self.logger.info(f"Added huggingface.co to NO_PROXY: {os.environ['NO_PROXY']}")
                    
                    # Create SSL context with relaxed verification rules
                    # This solves 'Basic Constraints not critical' error
                    custom_ssl_context = ssl.create_default_context()
                    custom_ssl_context.verify_flags &= ~ssl.VERIFY_X509_STRICT
                    custom_ssl_context.check_hostname = False
                    custom_ssl_context.verify_mode = ssl.CERT_NONE
                    
                    # Create custom HTTP adapter with our SSL context
                    class CustomHttpAdapter(requests.adapters.HTTPAdapter):
                        def init_poolmanager(self, *args, **kwargs):
                            kwargs['ssl_context'] = custom_ssl_context
                            kwargs['cert_reqs'] = 'CERT_NONE'
                            super().init_poolmanager(*args, **kwargs)
                        
                        def proxy_manager_for(self, *args, **kwargs):
                            kwargs['ssl_context'] = custom_ssl_context
                            kwargs['cert_reqs'] = 'CERT_NONE'
                            return super().proxy_manager_for(*args, **kwargs)
                    
                    # Create session factory that uses our custom adapter
                    def backend_factory() -> requests.Session:
                        session = requests.Session()
                        session.mount("https://", CustomHttpAdapter())
                        session.mount("http://", CustomHttpAdapter())
                        # Disable SSL verification completely
                        session.verify = False
                        return session
                    
                    # Configure huggingface_hub to use our custom backend
                    configure_http_backend(backend_factory=backend_factory)
                    
                    # Also set global SSL context as fallback
                    ssl._create_default_https_context = ssl._create_unverified_context
                    
                    self.logger.info("Configured huggingface_hub with custom SSL context for corporate environments")
                    
                except ImportError:
                    self.logger.warning("Could not configure huggingface_hub SSL settings - using fallback method")
                    # Fallback: global SSL context override
                    ssl._create_default_https_context = ssl._create_unverified_context
            
            try:
                # Initialize Lightning Whisper MLX with optimized settings
                self.model = LightningWhisperMLX(
                    model=self.model_name,
                    batch_size=self.batch_size,
                    quant=self.quantization  # None for best compatibility
                )
                
                self.logger.info(f"Lightning Whisper MLX loaded successfully")
                self.logger.info(f"Model: {self.model_name}, Batch size: {self.batch_size}")
                if self.quantization:
                    self.logger.info(f"Quantization: {self.quantization}")
                else:
                    self.logger.info("Quantization: None (full precision)")
                    
            except Exception as e:
                self.logger.error(f"Failed to load Lightning Whisper MLX: {e}")
                raise
    
    def convert_audio_format(self, input_path: str, output_path: str) -> None:
        """
        Convert audio file to WAV format suitable for Whisper.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output WAV file
        """
        try:
            self.logger.info(f"Converting {input_path} to WAV format")
            
            # Use pydub for format conversion
            audio = AudioSegment.from_file(input_path)
            
            # Convert to mono and set sample rate
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(self.sample_rate)
            
            # Export as WAV
            audio.export(output_path, format="wav")
            
            self.logger.info(f"Audio converted and saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to convert audio: {e}")
            raise
    
    def preprocess_audio(self, audio_path: str) -> str:
        """
        Preprocess audio to match vibe's exact specifications:
        - 16kHz sample rate
        - Mono (1 channel)
        - 16-bit PCM
        - FFmpeg loudnorm normalization (I=-16:TP=-1.5:LRA=11)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Path to preprocessed audio file
        """
        try:
            self.logger.debug("Preprocessing audio")
            
            # Check if audio already meets vibe specifications
            if self._should_normalize(audio_path):
                # Create normalized audio using FFmpeg with vibe's exact settings
                normalized_path = self._create_normalized_audio(audio_path)
                self.logger.debug("Audio preprocessing completed")
                return normalized_path
            else:
                self.logger.debug("Audio already in correct format")
                return audio_path
            
        except Exception as e:
            self.logger.error(f"Failed to preprocess audio: {e}")
            raise
    
    def _should_normalize(self, audio_path: str) -> bool:
        """
        Check if audio file needs normalization based on vibe's criteria.
        Returns False if the input is a WAV file with:
        - 1 channel (mono)
        - 16000 Hz sample rate  
        - 16 bits per sample
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            True if normalization is needed, False otherwise
        """
        try:
            # Check if it's a WAV file
            if not audio_path.lower().endswith('.wav'):
                return True
            
            # Load audio info without loading the full audio
            info = torchaudio.info(audio_path)
            
            # Check vibe's exact specifications
            if (info.num_channels == 1 and 
                info.sample_rate == 16000 and 
                info.bits_per_sample == 16):
                return False
            
            return True
            
        except Exception:
            # If we can't read the file info, assume normalization is needed
            return True
    
    def _create_normalized_audio(self, audio_path: str) -> str:
        """
        Create normalized audio using FFmpeg with vibe's exact settings:
        - Sample rate: 16000 Hz
        - Channels: 1 (mono)
        - Bit depth: 16-bit PCM signed little-endian
        - Loudness normalization: I=-16:TP=-1.5:LRA=11
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Path to normalized audio file
        """
        try:
            # Create temporary file for normalized audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                normalized_path = temp_file.name
            
            # FFmpeg command with vibe's exact normalization settings
            cmd = [
                "ffmpeg", "-y",  # Overwrite output file
                "-i", audio_path,  # Input file
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",  # Vibe's loudnorm settings
                "-ar", "16000",  # Sample rate: 16kHz
                "-ac", "1",      # Channels: mono
                "-c:a", "pcm_s16le",  # Codec: 16-bit PCM signed little-endian
                normalized_path
            ]
            
            # Run FFmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            self.logger.info(f"Audio normalized and saved to {normalized_path}")
            return normalized_path
            
        except Exception as e:
            self.logger.error(f"Failed to normalize audio: {e}")
            raise
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """
        Get audio duration in seconds using FFprobe.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return 0.0
    
    def _detect_silence_in_chunk(self, audio_path: str) -> bool:
        """
        Detect if audio chunk is mostly silent using FFmpeg silencedetect filter.
        
        Args:
            audio_path: Path to audio chunk
            
        Returns:
            True if chunk is mostly silent, False otherwise
        """
        try:
            # Use more sensitive silence detection
            cmd = [
                "ffmpeg", "-i", audio_path,
                "-af", "silencedetect=noise=-40dB:duration=1",
                "-f", "null", "-"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.STDOUT)
            
            # Count silence detections
            silence_count = result.stdout.count("silence_start")
            
            # Get chunk duration
            duration_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path]
            duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
            
            if duration_result.returncode == 0:
                chunk_duration = float(duration_result.stdout.strip())
                # If more than 70% is silence, consider it silent
                silence_ratio = silence_count * 1.0 / chunk_duration  # 1 second per silence detection
                return silence_ratio > 0.7
            
        except Exception as e:
            self.logger.debug(f"Silence detection failed: {e}")
        
        return False  # Default to processing if detection fails
    
    def _is_hallucination(self, text: str) -> bool:
        """
        Detect if transcribed text is likely a hallucination (repetitive patterns, common artifacts).
        
        Args:
            text: Transcribed text to check
            
        Returns:
            True if text appears to be hallucination, False otherwise
        """
        if not text or len(text.strip()) < 10:
            return True
        
        text = text.strip().lower()
        
        # Common hallucination patterns
        hallucination_patterns = [
            "повторяю", "повторю", "повтор",
            "спасибо за внимание", "спасибо за просмотр",
            "подписывайтесь", "ставьте лайки",
            "субтитры", "subtitles",
            "музыка", "music",
            "аплодисменты", "applause"
        ]
        
        # Check for repetitive patterns
        words = text.split()
        if len(words) > 5:
            # Check if more than 60% of words are the same
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_count = max(word_counts.values())
            if max_count / len(words) > 0.6:
                return True
        
        # Check for known hallucination patterns
        for pattern in hallucination_patterns:
            if pattern in text:
                return True
        
        # Check for very short repetitive segments
        if len(text) < 50 and len(set(words)) < 3:
            return True
        
        return False
    
    def _create_audio_chunk(self, audio_path: str, start_time: float, duration: float, output_path: str) -> str:
        """
        Create an audio chunk from the original file.
        
        Args:
            audio_path: Path to original audio file
            start_time: Start time in seconds
            duration: Duration of chunk in seconds
            output_path: Path for output chunk
            
        Returns:
            Path to created chunk
        """
        try:
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-ss", str(start_time),
                "-t", str(duration),
                "-c", "copy",
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg chunk creation failed: {result.stderr}")
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to create audio chunk: {e}")
            raise
    
    def transcribe_audio_chunked(self, audio_path: str, language: Optional[str] = None, output_path: Optional[str] = None) -> Dict:
        """
        Transcribe audio using chunked approach for live streaming effect.
        
        Args:
            audio_path: Path to audio file
            language: Language code or None for auto-detection
            output_path: Base path for live output
            
        Returns:
            Combined transcription result
        """
        try:
            if self.model is None:
                self.load_model()
            
            self.logger.info(f"Starting chunked transcription: {audio_path}")
            
            # Get audio duration
            total_duration = self._get_audio_duration(audio_path)
            if total_duration <= 0:
                # Fallback to regular transcription
                return self.transcribe_audio(audio_path, language)
            
            # Chunk settings (similar to whisper.cpp stream)
            chunk_duration = 30.0  # 30 seconds per chunk
            overlap_duration = 5.0  # 5 seconds overlap
            step_duration = chunk_duration - overlap_duration  # 25 seconds step
            
            print(f"\n🎵 Audio file: {os.path.basename(audio_path)}")
            print(f"⏱️  Duration: {total_duration:.1f} seconds")
            print(f"🧠 Model: {self.model_name}")
            print(f"🌍 Language: {language or 'auto'}")
            print(f"📦 Chunks: {chunk_duration:.0f}s with {overlap_duration:.0f}s overlap")
            print("\n🚀 Starting streaming transcription...\n")
            
            # Prepare live output file
            live_output_file = None
            if output_path:
                live_output_file = f"{output_path}_live.txt"
                # Clear the file
                with open(live_output_file, 'w', encoding='utf-8') as f:
                    f.write("")
            
            all_segments = []
            all_text_parts = []
            chunk_count = 0
            skipped_chunks = 0
            filtered_segments = 0
            
            # Process chunks
            current_time = 0.0
            while current_time < total_duration:
                chunk_count += 1
                
                # Calculate chunk boundaries
                chunk_end = min(current_time + chunk_duration, total_duration)
                actual_duration = chunk_end - current_time
                
                self.logger.debug(f"📦 Processing chunk {chunk_count}: {current_time:.1f}s - {chunk_end:.1f}s")
                
                # Create temporary chunk file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_chunk:
                    chunk_path = temp_chunk.name
                
                try:
                    # Extract chunk
                    self._create_audio_chunk(audio_path, current_time, actual_duration, chunk_path)
                    
                    # Check if chunk is mostly silent
                    if self._detect_silence_in_chunk(chunk_path):
                        self.logger.debug(f"🔇 Chunk {chunk_count} is mostly silent - skipping")
                        skipped_chunks += 1
                        # Clean up chunk file before continuing
                        if os.path.exists(chunk_path):
                            os.unlink(chunk_path)
                        current_time += step_duration
                        continue
                    
                    # Transcribe chunk
                    chunk_result = self.transcribe_audio(chunk_path, language)
                    
                    # Process chunk segments
                    if isinstance(chunk_result, dict) and "segments" in chunk_result:
                        for segment in chunk_result["segments"]:
                            # Adjust timestamps to global time
                            adjusted_segment = {
                                "start": segment["start"] + current_time,
                                "end": segment["end"] + current_time,
                                "text": segment["text"]
                            }
                            
                            # Skip overlapping segments from previous chunks
                            if not all_segments or adjusted_segment["start"] >= all_segments[-1]["end"] - 1.0:
                                # Filter out hallucinations
                                if self._is_hallucination(adjusted_segment["text"]):
                                    filtered_segments += 1
                                    timestamp = f"[{adjusted_segment['start']:.1f}s]"
                                    self.logger.debug(f"🚫 {timestamp} Filtered (hallucination): {adjusted_segment['text'][:50]}...")
                                    continue
                                
                                all_segments.append(adjusted_segment)
                                all_text_parts.append(adjusted_segment["text"])
                                
                                # Live output
                                timestamp = f"[{adjusted_segment['start']:.1f}s]"
                                print(f"{timestamp} {adjusted_segment['text']}")
                                
                                # Write to live file
                                if live_output_file:
                                    try:
                                        with open(live_output_file, 'w', encoding='utf-8') as f:
                                            f.write('\n'.join(all_text_parts))
                                    except Exception as e:
                                        self.logger.warning(f"Failed to write live output: {e}")
                                
                                # Small delay for visual effect
                                time.sleep(0.2)
                    
                finally:
                    # Clean up chunk file
                    if os.path.exists(chunk_path):
                        os.unlink(chunk_path)
                
                # Move to next chunk
                current_time += step_duration
            
            # Combine results
            combined_result = {
                "text": " ".join(all_text_parts),
                "segments": all_segments,
                "language": language or "ru"
            }
            
            print("\n" + "-" * 50)
            print(f"\n📊 Streaming transcription statistics:")
            print(f"   • Total chunks: {chunk_count}")
            print(f"   • Processed: {chunk_count - skipped_chunks}")
            print(f"   • Skipped (silence): {skipped_chunks}")
            print(f"   • Total segments: {len(all_segments) + filtered_segments}")
            print(f"   • Accepted: {len(all_segments)}")
            print(f"   • Filtered (hallucinations): {filtered_segments}")
            print(f"   • Total text: {len(combined_result['text'])} characters")
            
            self.logger.info("Chunked transcription completed")
            return combined_result
            
        except Exception as e:
            self.logger.error(f"Chunked transcription failed: {e}")
            # Fallback to regular transcription
            return self.transcribe_audio(audio_path, language)
    
    def transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """
        Transcribe audio using Lightning Whisper MLX.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'ru', 'es') or None for auto-detection
            
        Returns:
            Transcription result dictionary
        """
        try:
            if self.model is None:
                self.load_model()
            
            self.logger.debug(f"Transcribing audio with Lightning Whisper MLX: {audio_path}")
            
            # Preprocess audio to match vibe's specifications
            normalized_audio_path = self.preprocess_audio(audio_path)
            
            try:
                # Transcribe with Lightning Whisper MLX
                result = self.model.transcribe(
                    audio_path=normalized_audio_path,
                    language=language if language and language != "auto" else None
                )
                
                # Convert Lightning Whisper MLX format to standard format
                if isinstance(result, dict) and "segments" in result:
                    # Convert segments from [start_ms, end_ms, text] to dict format
                    if result["segments"] and isinstance(result["segments"][0], list):
                        converted_segments = []
                        for segment in result["segments"]:
                            if len(segment) >= 3:
                                converted_segments.append({
                                    "start": segment[0] / 1000.0,  # Convert ms to seconds
                                    "end": segment[1] / 1000.0,    # Convert ms to seconds
                                    "text": segment[2]
                                })
                        result["segments"] = converted_segments
                
                self.logger.debug("Lightning Whisper MLX transcription completed")
                return result
                
            finally:
                # Clean up temporary normalized audio file if it was created
                if normalized_audio_path != audio_path and os.path.exists(normalized_audio_path):
                    try:
                        os.unlink(normalized_audio_path)
                        self.logger.debug(f"Cleaned up temporary file: {normalized_audio_path}")
                    except Exception as cleanup_error:
                        self.logger.warning(f"Failed to clean up temporary file {normalized_audio_path}: {cleanup_error}")
            
        except Exception as e:
            self.logger.error(f"Failed to transcribe audio: {e}")
            raise
    
    def post_process_transcript(self, result: Dict) -> Dict:
        """
        Post-process transcription result for better formatting and quality.
        
        Args:
            result: Raw transcription result from Whisper
            
        Returns:
            Post-processed transcription result
        """
        try:
            self.logger.info("Post-processing transcript")
            
            # Extract segments and text
            segments = result.get("segments", [])
            text = result.get("text", "")
            
            # Clean and format text
            cleaned_text = self._clean_text(text)
            
            # Process segments for better formatting
            processed_segments = []
            for segment in segments:
                processed_segment = {
                    "id": segment.get("id"),
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "text": self._clean_text(segment.get("text", "")),
                    "confidence": segment.get("avg_logprob", 0.0)
                }
                processed_segments.append(processed_segment)
            
            # Create formatted transcript
            formatted_transcript = self._format_transcript(processed_segments)
            
            processed_result = {
                "text": cleaned_text,
                "segments": processed_segments,
                "formatted_transcript": formatted_transcript,
                "language": result.get("language"),
                "duration": max([s.get("end", 0) for s in segments]) if segments else 0
            }
            
            self.logger.info("Post-processing completed")
            return processed_result
            
        except Exception as e:
            self.logger.error(f"Failed to post-process transcript: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize transcribed text.
        
        Args:
            text: Raw transcribed text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common transcription issues
        text = re.sub(r'\s+([.!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # Ensure space after punctuation
        
        # Capitalize first letter of sentences
        sentences = re.split(r'([.!?])', text)
        cleaned_sentences = []
        for i, sentence in enumerate(sentences):
            if i % 2 == 0 and sentence.strip():  # Text parts (not punctuation)
                sentence = sentence.strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:]
                cleaned_sentences.append(sentence)
            else:
                cleaned_sentences.append(sentence)
        
        text = ''.join(cleaned_sentences)
        
        return text.strip()
    
    def _format_transcript(self, segments: List[Dict]) -> str:
        """
        Format transcript with timestamps for better readability.
        
        Args:
            segments: List of transcript segments
            
        Returns:
            Formatted transcript string
        """
        formatted_lines = []
        
        for segment in segments:
            start_time = self._format_timestamp(segment.get("start", 0))
            end_time = self._format_timestamp(segment.get("end", 0))
            text = segment.get("text", "").strip()
            
            if text:
                formatted_lines.append(f"[{start_time} -> {end_time}] {text}")
        
        return "\n".join(formatted_lines)
    
    def _format_plain_text(self, segments: List[Dict]) -> str:
        """
        Format transcript as readable paragraphs without timestamps.
        Groups segments into paragraphs for better readability.
        
        Args:
            segments: List of transcript segments
            
        Returns:
            Formatted plain text string
        """
        if not segments:
            return ""
        
        paragraphs = []
        current_paragraph = []
        
        for i, segment in enumerate(segments):
            text = segment.get("text", "").strip()
            if not text:
                continue
            
            # Clean up the text
            text = text.strip()
            if text:
                current_paragraph.append(text)
            
            # Create paragraph breaks every 3-4 segments or at natural breaks
            should_break = (
                len(current_paragraph) >= 3 and (
                    i == len(segments) - 1 or  # Last segment
                    len(current_paragraph) >= 4 or  # Max 4 segments per paragraph
                    text.endswith(('?', '!')) or  # Question/exclamation
                    text.endswith('...') or  # Pause
                    any(word in text.lower() for word in ['короче', 'ну вот', 'а я вот'])  # Topic change
                )
            )
            
            if should_break:
                if current_paragraph:
                    paragraph_text = " ".join(current_paragraph).strip()
                    paragraphs.append(paragraph_text)
                    current_paragraph = []
        
        # Add any remaining segments
        if current_paragraph:
            paragraph_text = " ".join(current_paragraph).strip()
            paragraphs.append(paragraph_text)
        
        return "\n\n".join(paragraphs)
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format timestamp in MM:SS format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def save_results(self, result: Dict, output_path: str, format_type: str = "all", ai_prompt_type: Optional[str] = None) -> None:
        """
        Save transcription results to file(s).
        
        Args:
            result: Processed transcription result
            output_path: Base output path (without extension)
            format_type: Output format ('txt', 'json', 'srt', 'all')
            ai_prompt_type: AI processing prompt type (meeting_notes, podcast_summary, etc.)
        """
        try:
            self.logger.info(f"Saving results to {output_path}")
            
            base_path = Path(output_path).with_suffix("")
            
            if format_type in ["txt", "all"]:
                # Save formatted plain text transcript (without timestamps)
                txt_path = base_path.with_suffix(".txt")
                formatted_plain_text = self._format_plain_text(result["segments"])
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(formatted_plain_text)
                self.logger.info(f"Formatted plain text saved to {txt_path}")
            
            if format_type in ["json", "all"]:
                # Save detailed JSON result
                json_path = base_path.with_suffix(".json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                self.logger.info(f"JSON result saved to {json_path}")
            
            if format_type in ["srt", "all"]:
                # Save SRT subtitle format
                srt_path = base_path.with_suffix(".srt")
                srt_content = self._generate_srt(result["segments"])
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)
                self.logger.info(f"SRT subtitles saved to {srt_path}")
            
            if format_type in ["formatted", "all"]:
                # Save formatted transcript with timestamps
                formatted_path = base_path.with_suffix(".formatted.txt")
                with open(formatted_path, "w", encoding="utf-8") as f:
                    f.write(result["formatted_transcript"])
                self.logger.info(f"Formatted transcript saved to {formatted_path}")
            
            # AI Processing
            if ai_prompt_type and self.enable_ai_processing and self.ai_processor:
                try:
                    self.logger.info(f"Processing transcript with AI using prompt: {ai_prompt_type}")
                    
                    # Use the plain text for AI processing
                    transcript_text = result["text"]
                    ai_result = self.ai_processor.process_transcript(transcript_text, ai_prompt_type)
                    
                    # Save AI processed result
                    ai_path = base_path.with_suffix(f".{ai_prompt_type}.md")
                    with open(ai_path, "w", encoding="utf-8") as f:
                        f.write(ai_result)
                    self.logger.info(f"AI processed result saved to {ai_path}")
                    
                except Exception as e:
                    self.logger.error(f"AI processing failed: {e}")
                    # Don't raise - continue with regular transcription
                
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise
    
    def _generate_srt(self, segments: List[Dict]) -> str:
        """
        Generate SRT subtitle format from segments.
        
        Args:
            segments: List of transcript segments
            
        Returns:
            SRT formatted string
        """
        srt_lines = []
        
        for i, segment in enumerate(segments, 1):
            start_time = self._format_srt_timestamp(segment.get("start", 0))
            end_time = self._format_srt_timestamp(segment.get("end", 0))
            text = segment.get("text", "").strip()
            
            if text:
                srt_lines.extend([
                    str(i),
                    f"{start_time} --> {end_time}",
                    text,
                    ""
                ])
        
        return "\n".join(srt_lines)
    
    def _format_srt_timestamp(self, seconds: float) -> str:
        """
        Format timestamp for SRT format (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            SRT formatted timestamp
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def process_file(self, input_path: str, output_path: str, language: Optional[str] = None, 
                    format_type: str = "all", ai_prompt_type: Optional[str] = None) -> Dict:
        """
        Process a single audio file through the complete pipeline.
        
        Args:
            input_path: Path to input audio file
            output_path: Base path for output files
            language: Language code or None for auto-detection
            format_type: Output format type
            ai_prompt_type: AI processing prompt type
            
        Returns:
            Processed transcription result
        """
        try:
            self.logger.info(f"Processing file: {input_path}")
            
            # Validate input file
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Create temporary WAV file if needed
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_wav_path = temp_file.name
            
            try:
                # Convert to WAV format
                self.convert_audio_format(input_path, temp_wav_path)
                
                # Transcribe audio with chunked approach for live streaming
                result = self.transcribe_audio_chunked(temp_wav_path, language, output_path)
                
                # Post-process transcript
                processed_result = self.post_process_transcript(result)
                
                # Save results
                self.save_results(processed_result, output_path, format_type, ai_prompt_type)
                
                self.logger.info("File processing completed successfully")
                return processed_result
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_wav_path):
                    os.unlink(temp_wav_path)
                
        except Exception as e:
            self.logger.error(f"Failed to process file: {e}")
            raise



