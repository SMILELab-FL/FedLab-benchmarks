#!/bin/bash
# bash scp_download.sh [smilelab09]
# $1 is
if $2
scp $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/config.py .
scp $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/utils.py .

scp $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/client.py .
scp $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/client_starter.py .
scp $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/start_clt.sh .

scp $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/server.py .
scp $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/server_starter.py .
scp $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/start_server.sh .

scp $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/standalone_main.py .
scp $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/start_standalone.sh .

