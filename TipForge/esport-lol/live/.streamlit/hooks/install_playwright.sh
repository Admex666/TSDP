#!/bin/bash
set -e

echo "Installing PlayWright browsers..."
python -m playwright install chromium --with-deps

echo "PlayWright installation complete!"