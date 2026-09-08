# EventVision AWS Deployment Guide

This document outlines how to deploy **EventVision** to Amazon Web Services (AWS) using EC2, S3, DynamoDB, and ECS.

## 1. Prerequisites
- AWS CLI configured (`aws configure`)
- Docker installed
- Python 3.12+

## 2. Setting Up AWS Services with CloudFormation
Run the included CloudFormation template to automatically provision S3, DynamoDB, and SQS:

```bash
aws cloudformation create-stack \
  --stack-name eventvision-infrastructure \
  --template-body file://aws/cloudformation.yaml \
  --region us-east-1
```

## 3. Configuration Update
Update `configs/config.yaml`:
```yaml
aws:
  enabled: true
  region: "us-east-1"
  s3_bucket: "eventvision-snapshots-<ACCOUNT_ID>-us-east-1"
  sqs_queue_url: "https://sqs.us-east-1.amazonaws.com/<ACCOUNT_ID>/eventvision-event-queue"
  dynamodb_table: "EventVisionEvents"
```

## 4. Building & Running Docker Image
Build the container:
```bash
docker build -t eventvision:latest .
```

Run locally or on AWS EC2:
```bash
docker run -d -p 8000:8000 -p 8501:8501 --name eventvision eventvision:latest
```
