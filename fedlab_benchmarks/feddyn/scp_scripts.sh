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

elif [[ "$2" == "upload" ]]; then
  echo "Upload scripts to $1"
  scp config.py $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/.
  scp utils.py $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/.

  scp client.py $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/.
  scp client_starter.py $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/.
  scp start_clt.sh $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/.

  scp server.py $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/.
  scp server_starter.py $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/.
  scp start_server.sh $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/.

  scp standalone_main.py $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/.
  scp start_standalone.sh $1:FedLab-benchmarks/fedlab_benchmarks/feddyn/.
else
  echo -e "the SECOND argument should be 'upload' or 'download'!!!!!!!\n"
fi
