import os

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["LOCAL_RANK"])
    print("rank=%d stage=start" % rank, flush=True)
    torch.cuda.set_device(rank)
    print("rank=%d stage=cuda device=%d" % (rank, torch.cuda.current_device()), flush=True)
    dist.init_process_group("gloo")
    print("rank=%d stage=process_group" % rank, flush=True)
    value = torch.tensor([float(rank + 1)], device="cpu")
    dist.all_reduce(value)
    print("rank=%d stage=cpu_all_reduce value=%s" % (rank, value.item()), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
