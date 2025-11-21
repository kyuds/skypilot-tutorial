#!/bin/bash
echo "Checking AWS credentials..."
echo ""

# Step 1: Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed"
    echo "Please install it: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

echo "✓ AWS CLI is installed"

# Step 2: Check if credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo ""
    echo "❌ AWS credentials are not configured or invalid"
    echo ""
    echo "Please configure your credentials using one of these methods:"
    echo "  1. Run: aws configure"
    echo "  2. Set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"
    echo "  3. Use IAM roles (if running on EC2)"
    exit 1
fi

echo "✓ AWS credentials are valid"

echo ""
echo "✅ Perfect! Let's do this 🔥"
