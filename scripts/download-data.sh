#!/bin/bash
set -e

BUCKET_NAME=""
REGION="us-east-1"

while [[ $# -gt 0 ]]; do
  case $1 in
    --bucket)
      BUCKET_NAME="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --bucket BUCKET_NAME    S3 bucket name (required)"
      echo "  --region REGION         AWS region (default: us-east-1)"
      echo "  -h, --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check if bucket name is provided
if [ -z "$BUCKET_NAME" ]; then
  echo "Error: Bucket name is required"
  echo ""
  echo "Usage: $0 --bucket BUCKET_NAME [--region REGION]"
  echo ""
  echo "Example:"
  echo "  $0 --bucket my-unique-bucket-name-12345"
  echo ""
  echo "Note: S3 bucket names must be globally unique across all AWS accounts."
  exit 1
fi

# echo "Creating S3 bucket: ${BUCKET_NAME} in ${REGION}..."
# aws s3 mb s3://${BUCKET_NAME} --region ${REGION}

echo "Downloading IMDb dataset from HuggingFace..."
TEMP_DIR=$(mktemp -d)
cd ${TEMP_DIR}

# Download the dataset files directly from HuggingFace
wget https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/train-00000-of-00001.parquet -O train.parquet
wget https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/test-00000-of-00001.parquet -O test.parquet

echo "Uploading to S3..."
aws s3 cp train.parquet s3://${BUCKET_NAME}/data/imdb/train.parquet --region ${REGION}
aws s3 cp test.parquet s3://${BUCKET_NAME}/data/imdb/test.parquet --region ${REGION}

echo "Cleaning up..."
cd -
rm -rf ${TEMP_DIR}

echo "Setup complete!"
echo "Bucket: s3://${BUCKET_NAME}"
echo "Training data: s3://${BUCKET_NAME}/data/imdb/train.parquet"
echo "Test data: s3://${BUCKET_NAME}/data/imdb/test.parquet"
