## To start ray cluster
# CUDA_VISIBLE_DEVICES=0,1,3,4,5,6 uv run ray start --head --port 6380 --num-gpus 6

## To get ray status
# uv run ray status --address=0.0.0.0:6380

## To stop ray cluster
# uv run ray stop