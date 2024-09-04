import gc
import torch

def print_gpu_utilization():
    if torch.cuda.is_available():
        used_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU 메모리 사용량: {used_memory:.3f} GB")
    else:
        print("런타임 유형을 GPU로 변경하세요")

def estimate_memory_of_gradients(model):
    total_memory = 0
    for param in model.parameters():
        if param.grad is not None:
            total_memory += param.grad.nelement() * param.grad.element_size()
    return total_memory

def estimate_memory_of_optimizer(optimizer):
    total_memory = 0
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                total_memory += v.nelement() * v.element_size()
    return total_memory

def cleanup():
    if 'model' in globals():
        del globals()['model']
    if 'dataset' in globals():
        del globals()['dataset']
    gc.collect()
    torch.cuda.empty_cache()