import os
import torch
import numpy as np

def save_checkpoint(model_dir, model, epoch, optimizer=None, save_name=None):
    """
    모델 체크포인트를 저장합니다.
    """
    if save_name is None:
        save_name = 'ckpt_ep{:03d}.pt'.format(epoch)
    
    state = {
        'model': model.state_dict(),
        'epoch': epoch
    }
    
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    
    torch.save(state, os.path.join(model_dir, save_name))
    print('모델 저장됨: {}'.format(os.path.join(model_dir, save_name)))

def load_checkpoint(model_dir, model, epoch=None, optimizer=None):
    """
    저장된 체크포인트를 로드합니다.
    """
    if epoch is None:
        path = os.path.join(model_dir, 'ckpt_latest.pt')
    else:
        path = os.path.join(model_dir, 'ckpt_ep{:03d}.pt'.format(epoch))
    
    if not os.path.exists(path):
        print('체크포인트를 찾을 수 없습니다!')
        return 0
    
    print('체크포인트 로드 중: {}'.format(path))
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint['epoch']

def cal_total_params(model):
    """
    모델의 총 파라미터 수를 계산합니다.
    """
    return sum(p.numel() for p in model.parameters())

def optimizer_to(optim, device):
    """
    옵티마이저의 상태를 특정 디바이스로 이동합니다.
    """
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)