#!/bin/bash
# bash scp_scripts.sh [smilelab09] [download/upload]
# $1 is
if [[ "$2" == "download" ]]; then
  echo "Download scripts from $1"
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

elif [[ "$2" == "upbashload" ]]; then
  echo "Upload scripts to $1"
  scp . $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/config.py
  scp . $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/utils.py

  scp . $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/client.py
  scp . $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/client_starter.py
  scp . $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/start_clt.sh

  scp . $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/server.py
  scp . $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/server_starter.py
  scp . $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/start_server.sh

  scp . $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/standalone_main.py
  scp . $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/start_standalone.sh
else
  echo -e "the SECOND argument should be 'upload' or 'download'!!!!!!!\n"
fi
