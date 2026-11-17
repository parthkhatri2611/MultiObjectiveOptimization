"""Explicit physical-GPU routing; never silently fall back to CPU."""
import os
import subprocess


def configure_embedding_device(protocol):
    policy = protocol.get('embedding_device', 'cpu')
    if policy == 'cpu':
        return 'cpu'
    if policy != 'cuda:1':
        raise ValueError('Supported embedding policies: cpu or cuda:1 (physical GPU 1)')
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True)
    rows = [line.split(',', 2) for line in result.stdout.splitlines() if line.strip()]
    selected = [row for row in rows if row[0].strip() == '1']
    if len(selected) != 1:
        raise RuntimeError('Physical GPU 1 is required; select Kaggle T4 x2')
    uuid = selected[0][1].strip()
    import torch
    if torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise RuntimeError('CUDA was initialized before GPU routing; use a fresh subprocess')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Physical GPU 1 is not available to the embedding process')
    torch.cuda.set_device(0)
    return 'cuda'


def device_diagnostics(protocol):
    record = {'embedding_policy': protocol.get('embedding_device', 'cpu'),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    if record['embedding_policy'] == 'cuda:1':
        import torch
        record.update(physical_gpu_index=1, logical_device='cuda:0',
                      gpu_name=torch.cuda.get_device_name(0),
                      allocated_bytes=torch.cuda.memory_allocated(0),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
    return record
