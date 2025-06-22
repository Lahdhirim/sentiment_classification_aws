import os
import boto3
from typing import Optional


class S3Manager:
    """
    A utility class for managing AWS S3 Service operations.
    """

    def __init__(self, bucket_name: str) -> None:
        """
        Initialize the S3Manager with the given bucket name.

        Args:
            bucket_name (str): Name of the S3 bucket to operate on.
        """
        self.bucket_name = bucket_name
        self.s3 = boto3.client('s3')

    def create_bucket_if_not_exists(self) -> None:
        """
        Create the S3 bucket if it does not already exist.
        """
        response = self.s3.list_buckets()
        existing_buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]

        if self.bucket_name not in existing_buckets:
            self.s3.create_bucket(Bucket=self.bucket_name)
            print(f"Bucket '{self.bucket_name}' created successfully.")
        else:
            print(f"Bucket '{self.bucket_name}' already exists. Reusing it.")

    def upload_directory(self, local_directory_path: str, s3_prefix: str) -> None:
        """
        Upload the contents of a local directory to the specified S3 prefix.

        Args:
            local_directory_path (str): Local directory path to upload.
            s3_prefix (str): Prefix in the S3 bucket under which files will be stored.
        """
        for root, _, files in os.walk(local_directory_path):
            for file in files:
                local_file_path = os.path.join(root, file).replace("\\", "/")
                s3_key = os.path.join(s3_prefix, file).replace("\\", "/")
                self.s3.upload_file(local_file_path, self.bucket_name, s3_key)
                print(f"Uploaded: {local_file_path} --> s3://{self.bucket_name}/{s3_key}")