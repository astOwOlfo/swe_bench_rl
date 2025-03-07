set -x

apt install -y curl git

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone https://github.com/astOwOlfo/swe_bench_rl.git --branch sf-compute

apt-get -y install cuda-toolkit-12-4
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}} && export LD_LIBRARY_PATH>
export CUDA_HOME=/usr/local/cuda-12.4
apt-get -y install python3.10-dev

cd swe_bench_rl

uv pip install setuptools psutil
uv pip install --no-build-isolation flash-attn==2.7.0.post2

