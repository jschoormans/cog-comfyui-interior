#!/bin/bash
echo "Starting ComfyUI..."
cd ComfyUI
python main.py --listen 0.0.0.0 --gpu-only
