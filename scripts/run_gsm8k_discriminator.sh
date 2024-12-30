# CUDA_VISIBLE_DEVICES=0 python run_src/do_discriminate.py \
#     --model_ckpt meta-llama/Llama-3.1-8B \
#     --root_dir run_outputs/GSM8K/Llama-3.1-8B \
#     --dataset_name GSM8K \
#     --note default
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_src/do_discriminate.py \
    --model_ckpt meta-llama/Llama-3.1-8B \
    --root_dir run_outputs/GSM8K/Llama-3.1-8B \
    --dataset_name GSM8K \
    --tensor_parallel_size 4 \
    --note tensor_parallelism