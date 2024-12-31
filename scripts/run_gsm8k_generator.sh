# CUDA_VISIBLE_DEVICES=0 python run_src/do_generate.py \
#     --dataset_name GSM8K \
#     --test_json_filename test_all \
#     --model_ckpt mistralai/Mistral-7B-v0.1 \
#     --note default \
#     --num_rollouts 16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python run_src/do_generate.py \
#     --model_ckpt meta-llama/Llama-3.1-8B \
#     --dataset_name GSM8K \
#     --tensor_parallel_size 4 \
#     --note tensor_parallelism \
#     --num_rollouts 16 \
#     --api vllm \
#     --model_parallel
# export NCCL_DEBUG=WARN
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1
export OMP_NUM_THREADS=16  # Optimize CPU usage
export CUDA_VISIBLE_DEVICES=0,1,2,3
#export NCCL_DEBUG=INFO

python run_src/do_generate.py \
    --model_ckpt meta-llama/Llama-3.1-8B \
    --dataset_name GSM8K \
    --note tensor_parallelism \
    --num_rollouts 16 \
    --api vllm \
    --model_parallel \
    --tensor_parallel_size 4 