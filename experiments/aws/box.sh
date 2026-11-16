#!/usr/bin/env bash
# Runs ON the EC2 box (started by aws_run.sh up/rerun). Executes the Colab notebook headless with
# papermill; outputs are saved after every cell, so ~/out.ipynb is always the latest state.
# Safe to rerun after a code fix: setup is skipped and experiments resume from results/.
set -uo pipefail
cd ~
rm -f ~/DONE
sudo shutdown -c 2>/dev/null; sudo shutdown -h +900 2>/dev/null   # hard cap: box stops after 15 h no matter what
if [ ! -d ~/venv ]; then
  sudo mkdir -p /content && sudo chown ubuntu:ubuntu /content    # notebook paths are Colab's /content/...
  sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ~/.local/bin/uv venv -q --seed --python 3.12 ~/venv            # Colab's python; notebook pins were checked on 3.12
  ~/.local/bin/uv pip install -q --python ~/venv/bin/python papermill ipykernel numpy pandas matplotlib \
      scipy python-dotenv nvidia-cudnn-cu12
  ~/venv/bin/python -m ipykernel install --user --name python3    # notebook kernelspec is "python3"
fi
source ~/venv/bin/activate                                       # so the notebook's `!pip` hits this venv
set -a; source ~/secrets.env; set +a
export PYTHONPATH=~/shim
sed -i 's/SMOKE = True /SMOKE = False/' ~/nb.ipynb               # full run
sed -i 's/E6\\": True/E6\\": False/' ~/nb.ipynb                  # user: skip E6 on AWS
papermill ~/nb.ipynb ~/out.ipynb -k python3 --cwd /content --log-output --no-progress-bar
echo "papermill exit $?" | tee ~/DONE
sudo shutdown -c; sudo shutdown -h +45                           # stops (not terminates) if nobody fetches
