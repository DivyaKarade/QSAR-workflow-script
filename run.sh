#!/bin/bash
docker run --gpus all --workdir /workspace -v "$(pwd)":/workspace chiral.sakuracr.jp/tensorflow:2025_12_02_v2 python QSAR_workflow_sample.py
