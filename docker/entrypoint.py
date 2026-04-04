"""
Fargate task entrypoint for MitraSETI cloud processing.

Reads an observation file from S3, runs the full MitraSETI pipeline
(Rust de-Doppler + GPU/Numba Taylor tree + chirp search + matched filter
+ ML classification), and writes results back to S3.

Environment variables:
    INPUT_BUCKET:   S3 bucket containing the input file
    INPUT_KEY:      S3 key of the .h5/.fil file
    RESULTS_BUCKET: S3 bucket for output results
    JOB_ID:         Unique job identifier
    MAX_DRIFT_RATE: Maximum drift rate (Hz/s), default 4.0
    MIN_SNR:        Minimum SNR threshold, default 10.0
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time

import boto3

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mitraseti-cloud")

s3 = boto3.client("s3")


def main():
    input_bucket = os.environ["INPUT_BUCKET"]
    input_key = os.environ["INPUT_KEY"]
    results_bucket = os.environ["RESULTS_BUCKET"]
    job_id = os.environ["JOB_ID"]

    logger.info("Job %s: processing s3://%s/%s", job_id, input_bucket, input_key)
    t_start = time.perf_counter()

    # Download file to local temp
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(input_key)[1], delete=False) as tmp:
        local_path = tmp.name
        s3.download_file(input_bucket, input_key, local_path)
        logger.info("Downloaded to %s (%.1f MB)", local_path, os.path.getsize(local_path) / 1e6)

    sys.path.insert(0, "/app")
    from pipeline import MitraSETIPipeline

    pipeline = MitraSETIPipeline()

    try:
        result = pipeline.process_file(local_path)
    except Exception as e:
        logger.error("Pipeline crashed: %s", e, exc_info=True)
        elapsed = time.perf_counter() - t_start
        output = {
            "job_id": job_id,
            "input_file": f"s3://{input_bucket}/{input_key}",
            "processing_time_seconds": round(elapsed, 2),
            "status": "error",
            "error": str(e),
            "pipeline_version": "0.3.0",
        }
        results_key = f"jobs/{job_id}/results.json"
        s3.put_object(
            Bucket=results_bucket, Key=results_key,
            Body=json.dumps(output, default=str, indent=2),
            ContentType="application/json",
        )
        _update_job_status(job_id, "error", output)
        os.unlink(local_path)
        logger.info("Job %s failed: %s (%.1fs)", job_id, e, elapsed)
        return

    elapsed = time.perf_counter() - t_start

    summary = result.get("summary", {})
    candidates = result.get("candidates", [])

    output = {
        "job_id": job_id,
        "input_file": f"s3://{input_bucket}/{input_key}",
        "processing_time_seconds": round(elapsed, 2),
        "status": summary.get("status", "complete"),
        "raw_hits": summary.get("total_hits_raw", 0),
        "filtered_hits": summary.get("total_hits_filtered", 0),
        "candidate_count": summary.get("candidate_count", 0),
        "anomaly_count": summary.get("anomaly_count", 0),
        "rfi_count": summary.get("rfi_count", 0),
        "top_candidates": sorted(
            candidates, key=lambda c: c.get("snr", 0), reverse=True
        )[:20],
        "timing": result.get("timing", {}),
        "pipeline_version": "0.3.0",
    }

    # Write results to S3
    results_key = f"jobs/{job_id}/results.json"
    s3.put_object(
        Bucket=results_bucket,
        Key=results_key,
        Body=json.dumps(output, default=str, indent=2),
        ContentType="application/json",
    )

    # Update job status in DynamoDB
    _update_job_status(job_id, "complete", output)

    # Cleanup
    os.unlink(local_path)

    logger.info(
        "Job %s complete: %d candidates in %.1fs",
        job_id, output["candidate_count"], elapsed,
    )


def _update_job_status(job_id: str, status: str, result: dict):
    """Update job record in DynamoDB."""
    table_name = os.environ.get("JOBS_TABLE", "mitraseti-jobs")
    try:
        dynamodb = boto3.resource("dynamodb")
        table = dynamodb.Table(table_name)
        table.update_item(
            Key={"job_id": job_id},
            UpdateExpression="SET #s = :s, completed_at = :t, results = :r",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":s": status,
                ":t": int(time.time()),
                ":r": json.dumps(result, default=str),
            },
        )
    except Exception as e:
        logger.warning("Could not update DynamoDB: %s", e)


if __name__ == "__main__":
    main()
