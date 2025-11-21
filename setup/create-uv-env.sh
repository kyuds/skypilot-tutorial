#!/bin/bash
echo "Creating UV environment with Python 3.11..."
echo ""

# Step 1: Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed"
    echo ""
    echo "Install UV with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo "Or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Step 2: Create the environment
echo "Creating environment with Python 3.11..."
uv venv --seed --python 3.11

echo ""
echo "✅ Environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
