#!/bin/bash

# Clean up old build files
echo "🧹 Cleaning up old builds..."
rm -rf build/ dist/ *.egg-info/

# Set up the test environment
echo "🔧 Setting up test environment..."
python -m pip install --upgrade pip
python -m pip install --upgrade build twine

# Build the package
echo "📦 Building package..."
python -m build

# Upload the package to PyPI
echo "🚀 Uploading to PyPI..."
python -m twine upload dist/*

echo "✨ Done! Package published successfully!"
