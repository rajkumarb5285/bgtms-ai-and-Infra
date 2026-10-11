"""
BGTMS - Amazon S3 Dataset Acquisition & Sync Tool (M1-05)
Bucket: s3://bgtms-ai-datasets/ (Region: ap-south-1)
"""

import os
import sys
import argparse
import boto3
from botocore.exceptions import ClientError, ProfileNotFound


def get_s3_client(profile_name="roadsense", region="ap-south-1"):
    try:
        session = boto3.Session(profile_name=profile_name, region_name=region)
        return session.client("s3")
    except ProfileNotFound:
        print(f"[!] Profile '{profile_name}' not found, falling back to default session")
        session = boto3.Session(region_name=region)
        return session.client("s3")
    except Exception as e:
        print(f"[!] Error creating boto3 session: {e}")
        return boto3.client("s3", region_name=region)


def list_bucket_contents(bucket_name="bgtms-ai-datasets", prefix="raw/", profile_name="roadsense"):
    s3 = get_s3_client(profile_name=profile_name)
    print(f"[*] Listing objects in s3://{bucket_name}/{prefix} ...")
    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        objects = []
        total_size = 0
        for page in pages:
            for item in page.get("Contents", []):
                size_mb = item["Size"] / (1024 * 1024)
                objects.append({
                    "key": item["Key"],
                    "size_bytes": item["Size"],
                    "size_mb": size_mb,
                    "last_modified": str(item["LastModified"])
                })
                total_size += item["Size"]
                
        print(f"[+] Found {len(objects)} files (Total: {total_size / (1024**3):.2f} GB):")
        for obj in objects:
            print(f"    - {obj['key']} ({obj['size_mb']:.1f} MB, {obj['last_modified']})")
        return objects
    except ClientError as e:
        print(f"[ERROR] Failed to list s3://{bucket_name}: {e}")
        return []


def download_file(bucket_name, key, local_dest, profile_name="roadsense"):
    s3 = get_s3_client(profile_name=profile_name)
    os.makedirs(os.path.dirname(local_dest), exist_ok=True)
    
    print(f"[*] Downloading s3://{bucket_name}/{key} -> {local_dest} ...")
    try:
        head = s3.head_object(Bucket=bucket_name, Key=key)
        total_bytes = head["ContentLength"]
        downloaded = 0
        last_percent = -1

        def progress_callback(bytes_amount):
            nonlocal downloaded, last_percent
            downloaded += bytes_amount
            percent = int((downloaded / total_bytes) * 100) if total_bytes > 0 else 100
            if percent != last_percent and percent % 10 == 0:
                print(f"    -> Progress: {percent}% ({downloaded / (1024*1024):.1f}/{total_bytes / (1024*1024):.1f} MB)")
                last_percent = percent

        s3.download_file(
            Bucket=bucket_name,
            Key=key,
            Filename=local_dest,
            Callback=progress_callback
        )
        print(f"[+] Successfully downloaded {key} ({total_bytes / (1024*1024):.1f} MB)")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {key}: {e}")
        return False


def upload_file(bucket_name, local_path, s3_key, profile_name="roadsense"):
    s3 = get_s3_client(profile_name=profile_name)
    if not os.path.exists(local_path):
        print(f"[ERROR] Local file not found: {local_path}")
        return False

    file_size = os.path.getsize(local_path)
    print(f"[*] Uploading {local_path} ({file_size / (1024*1024):.1f} MB) -> s3://{bucket_name}/{s3_key} ...")
    uploaded = 0
    last_percent = -1

    def progress_callback(bytes_amount):
        nonlocal uploaded, last_percent
        uploaded += bytes_amount
        percent = int((uploaded / file_size) * 100) if file_size > 0 else 100
        if percent != last_percent and percent % 20 == 0:
            print(f"    -> Progress: {percent}% ({uploaded / (1024*1024):.1f}/{file_size / (1024*1024):.1f} MB)")
            last_percent = percent

    try:
        s3.upload_file(
            Filename=local_path,
            Bucket=bucket_name,
            Key=s3_key,
            Callback=progress_callback
        )
        print(f"[+] Successfully uploaded s3://{bucket_name}/{s3_key}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to upload {local_path} to s3://{bucket_name}/{s3_key}: {e}")
        return False


def zip_directory(dir_path, output_zip_path):
    import zipfile
    print(f"[*] Compressing {dir_path} -> {output_zip_path} ...")
    os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(dir_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, start=dir_path)
                zipf.write(full_path, arcname=rel_path)
    zip_size = os.path.getsize(output_zip_path) / (1024 * 1024)
    print(f"[+] Created archive {output_zip_path} ({zip_size:.1f} MB)")
    return output_zip_path


def main():
    parser = argparse.ArgumentParser(description="BGTMS S3 Dataset & Model Sync")
    parser.add_argument("--bucket", default="bgtms-ai-datasets", help="S3 bucket name")
    parser.add_argument("--profile", default="roadsense", help="AWS CLI profile name")
    parser.add_argument("--list", action="store_true", help="List files in S3")
    parser.add_argument("--download-target", choices=["all", "indian-vehicle", "idd", "ua-detrac"],
                        default=None, help="Target dataset to download")
    parser.add_argument("--dest-dir", default="data/raw", help="Local destination directory")
    parser.add_argument("--upload-model", action="store_true", help="Upload trained model weights and report to S3")
    parser.add_argument("--upload-extracted", action="store_true", help="Upload extracted dataset archive to S3")
    parser.add_argument("--upload-processed", action="store_true", help="Upload processed dataset to S3")
    parser.add_argument("--upload-all", action="store_true", help="Upload models, extracted, and processed data to S3")
    args = parser.parse_args()

    if args.list:
        list_bucket_contents(args.bucket, prefix="", profile_name=args.profile)
        return

    if args.upload_model or args.upload_all:
        model_path = "bgtms/models/vehicle_detector.pth"
        if os.path.exists(model_path):
            upload_file(args.bucket, model_path, "models/vehicle_detector.pth", profile_name=args.profile)
        report_path = "docs/M1-10_detector_training_report.md"
        if os.path.exists(report_path):
            upload_file(args.bucket, report_path, "models/M1-10_detector_training_report.md", profile_name=args.profile)
        curve_path = "data/processed/training_loss_curve.png"
        if os.path.exists(curve_path):
            upload_file(args.bucket, curve_path, "models/training_loss_curve.png", profile_name=args.profile)

    if args.upload_extracted or args.upload_all:
        ext_dir = "data/extracted/indian-vehicle-dataset"
        if os.path.exists(ext_dir):
            zip_dest = "data/extracted_indian_vehicle.zip"
            zip_directory(ext_dir, zip_dest)
            upload_file(args.bucket, zip_dest, "extracted/extracted_indian_vehicle.zip", profile_name=args.profile)

    if args.upload_processed or args.upload_all:
        proc_dir = "data/processed"
        if os.path.exists(proc_dir):
            zip_dest = "data/processed_dataset.zip"
            zip_directory(proc_dir, zip_dest)
            upload_file(args.bucket, zip_dest, "processed/processed_dataset.zip", profile_name=args.profile)
        manifest_path = "data/splits/split_manifest.json"
        if os.path.exists(manifest_path):
            upload_file(args.bucket, manifest_path, "processed/split_manifest.json", profile_name=args.profile)

    if args.download_target:
        targets = {
            "indian-vehicle": "raw/indian-vehicle/indian-vehicle-dataset.zip",
            "idd": "raw/idd/indian-driving-dataset.zip",
            "ua-detrac": "raw/ua-detrac/ua-detrac-orig.zip"
        }
        keys = list(targets.values()) if args.download_target == "all" else [targets[args.download_target]]
        for key in keys:
            local_path = os.path.join(args.dest_dir, os.path.basename(key))
            download_file(args.bucket, key, local_path, profile_name=args.profile)


if __name__ == "__main__":
    main()
