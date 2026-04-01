#!/bin/bash
# CLIPFORGE — Local Setup Script
# Run this once to set up, then again to start the server

set -e

echo ""
echo "══════════════════════════════════════"
echo "  CLIPFORGE Local Backend Setup"
echo "══════════════════════════════════════"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install from https://python.org"
    exit 1
fi
echo "✅ Python: $(python3 --version)"

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo ""
    echo "❌ FFmpeg not found. Install it:"
    echo ""
    echo "  macOS:   brew install ffmpeg"
    echo "  Ubuntu:  sudo apt install ffmpeg"
    echo "  Windows: https://ffmpeg.org/download.html"
    echo ""
    exit 1
fi
echo "✅ FFmpeg: $(ffmpeg -version 2>&1 | head -1)"

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Install deps
echo "📦 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "══════════════════════════════════════"
echo "  Starting CLIPFORGE backend..."
echo "  Open index.html in your browser"
echo "  Enter: http://localhost:8000"
echo "══════════════════════════════════════"
echo ""

# Override dirs to local for development
export CLIPFORGE_LOCAL=1

uvicorn main:app --reload --host 0.0.0.0 --port 8000
