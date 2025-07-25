#!/bin/bash

# Script to convert Whisper large-v3-turbo to MLX format for Lightning Whisper MLX
# This creates the weights.npz format needed by Lightning Whisper MLX

set -e

echo "🔄 Converting Whisper large-v3-turbo to MLX format..."

# Create models directory if it doesn't exist
mkdir -p models

# Clone MLX Examples repo if not exists
if [ ! -d "mlx-examples" ]; then
    echo "📥 Cloning MLX Examples repository..."
    git clone https://github.com/ml-explore/mlx-examples.git
fi

# Navigate to whisper conversion directory
cd mlx-examples/whisper

# Install required dependencies
echo "📦 Installing conversion dependencies..."
pip install -r requirements.txt

# Convert the model
echo "⚙️ Converting large-v3-turbo model..."
python convert.py \
    --torch-name-or-path large-v3-turbo \
    --mlx-path ../../models/large-v3-turbo \
    --dtype float16

echo "✅ Conversion complete!"
echo "📁 Model saved to: models/large-v3-turbo/"
echo "📄 Files created:"
echo "   - models/large-v3-turbo/weights.npz"
echo "   - models/large-v3-turbo/config.json"

# Go back to project root
cd ../..

echo ""
echo "🚀 Now you can use the model with Lightning Whisper MLX:"
echo "   ./audia audio.m4a -m models/large-v3-turbo"
