#!/bin/bash

IMAGE=gcr.io/morphgpt/dataproc_serverless_container:1.0.1

# Download the BigQuery connector.
# gsutil cp \
#   gs://spark-lib/bigquery/spark-bigquery-with-dependencies_2.12-0.22.2.jar .

# Download the Miniconda3 installer.
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh


# Build and push the image.
docker build -t "${IMAGE}" .
docker push "${IMAGE}"