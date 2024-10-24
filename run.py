import os
import time
import numpy as np


def gpu_info(gpu_index):
    info = os.popen('nvidia-smi|grep %').read().split('\n')[gpu_index].split('|')
    memory = int(info[2].split('/')[0].strip()[:-3])
    return memory



if __name__ == "__main__":
    gpu_device = 4
    
    while True:
        memory = gpu_info(gpu_device)

        if memory < 1000:
            break
        time.sleep(10)
        print("waiting | gpu ", str(gpu_device), " mem is ", memory)

    print("GPU", gpu_device, " is free, now gpu memory is only", memory)

    
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_device} /home/u2024001042/miniconda3/envs/mrag/bin/python /home/u2024001042/dev/We-Math/gen_vote.py" 
    os.system(cmd)



