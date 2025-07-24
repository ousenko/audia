#!/usr/bin/env python3
"""
AI Processor for transcript analysis using OpenAI-compatible APIs.

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
"""

import os
import yaml
import json
import requests
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


@dataclass
class AIProcessingConfig:
    """Configuration for AI processing."""
    api_url: str
    api_key: str
    model: str
    temperature: float = 0.3
    max_tokens: int = 4000
    timeout: int = 60


class AIProcessor:
    """Handles AI-powered transcript processing using OpenAI-compatible APIs."""
    
    def __init__(self, config_path: str = "config.yaml", require_api_key: bool = True):
        """Initialize AI processor with configuration."""
        self.logger = logging.getLogger(__name__)
        
        # Load .env file if available
        if DOTENV_AVAILABLE:
            env_path = Path(".env")
            if env_path.exists():
                load_dotenv(env_path)
                self.logger.info("Loaded environment variables from .env file")
        
        try:
            self.config = self._load_config(config_path, require_api_key)
            self.prompts_dir = Path("prompts")
            self.logger.info(f"AI processor initialized with model: {self.config.model}")
        except Exception as e:
            if require_api_key:
                self.logger.error(f"Failed to initialize AI processor: {e}")
                raise
            else:
                # Allow initialization without API key for prompt listing
                self.config = None
                self.prompts_dir = Path("prompts")
                self.logger.info("AI processor initialized without API key (prompt listing only)")
    
    def _load_config(self, config_path: str, require_api_key: bool = True) -> AIProcessingConfig:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            ai_config = config_data.get('ai_processing', {})
            
            # Get API settings from environment variables (loaded from .env)
            api_url = os.getenv('OPENAI_API_URL', 'https://api.openai.com/v1/chat/completions')
            api_key = os.getenv('OPENAI_API_KEY', '')
            model = os.getenv('OPENAI_MODEL', ai_config.get('model', 'gpt-4o-mini'))
            
            # Get optional settings from environment or config
            temperature = float(os.getenv('OPENAI_TEMPERATURE', ai_config.get('temperature', 0.3)))
            max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', ai_config.get('max_tokens', 4000)))
            timeout = int(os.getenv('OPENAI_TIMEOUT', ai_config.get('timeout', 60)))
            
            if require_api_key and not api_key:
                raise ValueError("API key not found. Set OPENAI_API_KEY in .env file or environment variable")
            
            return AIProcessingConfig(
                api_url=api_url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            raise
    
    def load_prompt(self, prompt_type: str) -> str:
        """
        Load prompt template from file.
        
        Args:
            prompt_type: Type of prompt (meeting_notes, podcast_summary, etc.)
            
        Returns:
            Prompt template string
        """
        prompt_file = self.prompts_dir / f"{prompt_type}.txt"
        
        if not prompt_file.exists():
            available_prompts = [f.stem for f in self.prompts_dir.glob("*.txt")]
            raise FileNotFoundError(
                f"Prompt file '{prompt_file}' not found. "
                f"Available prompts: {', '.join(available_prompts)}"
            )
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            self.logger.error(f"Failed to load prompt from {prompt_file}: {e}")
            raise
    
    def _chunk_transcript(self, transcript: str, chunk_size: int = 8000) -> List[str]:
        """
        Split long transcript into chunks for processing.
        
        Args:
            transcript: Full transcript text
            chunk_size: Maximum characters per chunk
            
        Returns:
            List of transcript chunks
        """
        if len(transcript) <= chunk_size:
            return [transcript]
        
        chunks = []
        words = transcript.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _make_api_request(self, messages: List[Dict[str, str]]) -> str:
        """
        Make API request to OpenAI-compatible endpoint.
        
        Args:
            messages: List of message objects for the API
            
        Returns:
            Generated response text
        """
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        try:
            self.logger.info(f"Making API request to {self.config.api_url}")
            response = requests.post(
                self.config.api_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise
        except (KeyError, IndexError) as e:
            self.logger.error(f"Unexpected API response format: {e}")
            raise
    
    def process_transcript(
        self, 
        transcript: str, 
        prompt_type: str = "meeting_notes",
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        Process transcript using AI with specified prompt.
        
        Args:
            transcript: Transcript text to process
            prompt_type: Type of prompt to use
            custom_prompt: Custom prompt string (overrides prompt_type)
            
        Returns:
            Processed result from AI
        """
        try:
            # Load prompt template
            if custom_prompt:
                prompt_template = custom_prompt
            else:
                prompt_template = self.load_prompt(prompt_type)
            
            # Fill prompt with transcript
            full_prompt = prompt_template.format(transcript=transcript)
            
            # Check if transcript needs chunking
            chunks = self._chunk_transcript(transcript)
            
            if len(chunks) == 1:
                # Single request for short transcripts
                messages = [
                    {"role": "user", "content": full_prompt}
                ]
                return self._make_api_request(messages)
            
            else:
                # Multi-chunk processing for long transcripts
                self.logger.info(f"Processing transcript in {len(chunks)} chunks")
                
                chunk_results = []
                for i, chunk in enumerate(chunks, 1):
                    self.logger.info(f"Processing chunk {i}/{len(chunks)}")
                    
                    chunk_prompt = prompt_template.format(transcript=chunk)
                    messages = [
                        {"role": "user", "content": chunk_prompt}
                    ]
                    
                    chunk_result = self._make_api_request(messages)
                    chunk_results.append(f"## Часть {i}\n\n{chunk_result}")
                
                # Combine results
                combined_result = "\n\n".join(chunk_results)
                
                # Final synthesis if multiple chunks
                synthesis_prompt = f"""
Объедини следующие части анализа в единое, связное резюме. 
Убери дублирование, создай общую структуру и сделай текст целостным:

{combined_result}
"""
                
                messages = [
                    {"role": "user", "content": synthesis_prompt}
                ]
                
                return self._make_api_request(messages)
                
        except Exception as e:
            self.logger.error(f"Failed to process transcript: {e}")
            raise
    
    def list_available_prompts(self) -> List[str]:
        """
        Get list of available prompt types.
        
        Returns:
            List of available prompt names
        """
        if not self.prompts_dir.exists():
            return []
        
        return [f.stem for f in self.prompts_dir.glob("*.txt")]



