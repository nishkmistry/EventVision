import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("AWSCloudAdapter")

class AWSCloudAdapter:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        aws_cfg = self.config.get("aws", {})
        self.enabled = aws_cfg.get("enabled", False)
        self.region = aws_cfg.get("region", "us-east-1")
        self.s3_bucket = aws_cfg.get("s3_bucket", "eventvision-snapshots-bucket")
        self.s3_client = None

        if self.enabled:
            try:
                import boto3
                self.s3_client = boto3.client('s3', region_name=self.region)
            except Exception as e:
                logger.warning(f"AWS Boto3 initialization failed: {e}. Falling back to offline AWS cloud mock mode.")

    def upload_snapshot_to_s3(self, local_file_path: str, s3_key: Optional[str] = None) -> Dict[str, Any]:
        if not s3_key:
            s3_key = f"snapshots/{os.path.basename(local_file_path)}"

        if self.enabled and self.s3_client:
            try:
                self.s3_client.upload_file(local_file_path, self.s3_bucket, s3_key)
                s3_url = f"https://{self.s3_bucket}.s3.{self.region}.amazonaws.com/{s3_key}"
                return {"uploaded": True, "s3_url": s3_url, "mode": "AWS_S3"}
            except Exception as e:
                logger.error(f"Failed to upload {local_file_path} to S3: {e}")
                return {"uploaded": False, "error": str(e), "mode": "AWS_S3_ERROR"}

        return {
            "uploaded": True,
            "s3_url": f"s3://{self.s3_bucket}/{s3_key}",
            "mode": "MOCK_CLOUD"
        }
