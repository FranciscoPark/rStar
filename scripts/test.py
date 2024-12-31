import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"Rank {dist.get_rank()} is using GPU {torch.cuda.current_device()}")

if __name__ == "__main__":
    main()

