"""
airbench94_modified.py
======================

Modified version of airbench94.py with experimental modifications that can be toggled via command-line arguments.

Based on:
- Original: https://github.com/KellerJordan/cifar10-airbench/blob/master/legacy/airbench94.py

Paper References:
- Gradient Centralization: Yong et al., ECCV 2020 - https://arxiv.org/abs/2004.01461
- Anti-Aliased Pooling: Zhang, ICML 2019 - https://arxiv.org/abs/1904.11486
- Mish Activation: Misra, 2019 - https://arxiv.org/abs/1908.08681
- Mixup: Zhang et al., ICLR 2018 - https://arxiv.org/abs/1710.09412
"""

#############################################
#            Setup/Hyperparameters          #
#############################################

import os
import sys
import uuid
import argparse
import json
from math import ceil
from datetime import datetime
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Speedrun with Modifications')
    
    # Experiment settings
    parser.add_argument('--n_runs', type=int, default=25,
                        help='Number of training runs (default: 25)')
    parser.add_argument('--seed_start', type=int, default=0,
                        help='Starting random seed (default: 0)')
    parser.add_argument('--output_dir', type=str, default='logs',
                        help='Output directory for logs (default: logs)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment (auto-generated if not provided)')
    
    # Modification toggles
    parser.add_argument('--gradient_centralization', action='store_true',
                        help='Enable gradient centralization (Yong et al., ECCV 2020)')
    parser.add_argument('--antialiased_pooling', action='store_true',
                        help='Enable anti-aliased pooling (Zhang, ICML 2019)')
    parser.add_argument('--blur_filter_size', type=int, default=3, choices=[3, 5],
                        help='Blur kernel size for anti-aliased pooling (default: 3)')
    parser.add_argument('--mish_activation', action='store_true',
                        help='Replace GELU with Mish activation')
    parser.add_argument('--bn_momentum_schedule', action='store_true',
                        help='Enable BatchNorm momentum scheduling')
    parser.add_argument('--bn_momentum_start', type=float, default=0.8,
                        help='Starting BN momentum (default: 0.8)')
    parser.add_argument('--bn_momentum_end', type=float, default=0.4,
                        help='Ending BN momentum (default: 0.4)')
    parser.add_argument('--mixup', action='store_true',
                        help='Enable mixup augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='Mixup alpha parameter (default: 0.2)')
    parser.add_argument('--label_smoothing_schedule', action='store_true',
                        help='Enable label smoothing scheduling')
    parser.add_argument('--label_smoothing_start', type=float, default=0.3,
                        help='Starting label smoothing (default: 0.3)')
    parser.add_argument('--label_smoothing_end', type=float, default=0.1,
                        help='Ending label smoothing (default: 0.1)')
    
    return parser.parse_args()

# Parse args early so MODIFICATIONS dict is available for class definitions
ARGS = parse_args()

MODIFICATIONS = {
    'gradient_centralization': ARGS.gradient_centralization,
    'antialiased_pooling': ARGS.antialiased_pooling,
    'blur_filter_size': ARGS.blur_filter_size,
    'mish_activation': ARGS.mish_activation,
    'bn_momentum_schedule': ARGS.bn_momentum_schedule,
    'bn_momentum_start': ARGS.bn_momentum_start,
    'bn_momentum_end': ARGS.bn_momentum_end,
    'mixup': ARGS.mixup,
    'mixup_alpha': ARGS.mixup_alpha,
    'label_smoothing_schedule': ARGS.label_smoothing_schedule,
    'label_smoothing_start': ARGS.label_smoothing_start,
    'label_smoothing_end': ARGS.label_smoothing_end,
}

hyp = {
    'opt': {
        'train_epochs': 9.9,
        'batch_size': 1024,
        'lr': 11.5,
        'momentum': 0.85,
        'weight_decay': 0.0153,
        'bias_scaler': 64.0,
        'label_smoothing': 0.2,
        'whiten_bias_epochs': 3,
    },
    'aug': {
        'flip': True,
        'translate': 2,
    },
    'net': {
        'widths': {
            'block1': 64,
            'block2': 256,
            'block3': 256,
        },
        'batchnorm_momentum': 0.6,
        'scaling_factor': 1/9,
        'tta_level': 2,
    },
}

#############################################
#        MODIFICATION IMPLEMENTATIONS       #
#############################################

# --- Gradient Centralization ---
def centralize_gradient(grad):
    """Centralize gradient to have zero mean."""
    if grad is None:
        return grad
    if len(grad.shape) > 1:
        grad.add_(-grad.mean(dim=tuple(range(1, len(grad.shape))), keepdim=True))
    return grad

class SGD_GC(torch.optim.SGD):
    """SGD with Gradient Centralization."""
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and len(p.grad.shape) > 1:
                    centralize_gradient(p.grad)
        return super().step(closure)

# --- Anti-Aliased Pooling ---
class BlurPool2d(nn.Module):
    """Blur pooling layer for anti-aliasing."""
    def __init__(self, channels, filter_size=3, stride=2):
        super().__init__()
        self.channels = channels
        self.stride = stride
        
        if filter_size == 3:
            kernel = torch.tensor([1., 2., 1.])
        elif filter_size == 5:
            kernel = torch.tensor([1., 4., 6., 4., 1.])
        else:
            kernel = torch.tensor([1., 2., 1.])
        
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel[None, None, :, :].repeat(channels, 1, 1, 1))
        self.pad = (filter_size - 1) // 2
        
    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        return F.conv2d(x, self.kernel.to(x.dtype), stride=self.stride, groups=self.channels)

class MaxBlurPool2d(nn.Module):
    """MaxPool followed by BlurPool."""
    def __init__(self, channels, filter_size=3):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.blurpool = BlurPool2d(channels, filter_size=filter_size, stride=2)
        
    def forward(self, x):
        return self.blurpool(self.maxpool(x))

# --- Mish Activation ---
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# --- Mixup ---
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None, drop_last=None, shuffle=None, gpu=0):
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device(gpu))
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {}
        self.epoch = 0
        self.aug = aug or {}
        self.batch_size = batch_size
        self.drop_last = train if drop_last is None else drop_last
        self.shuffle = train if shuffle is None else shuffle

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):
        if self.epoch == 0:
            images = self.proc_images['norm'] = self.normalize(self.images)
            if self.aug.get('flip', False):
                images = self.proc_images['flip'] = batch_flip_lr(images)
            pad = self.aug.get('translate', 0)
            if pad > 0:
                self.proc_images['pad'] = F.pad(images, (pad,)*4, 'reflect')

        if self.aug.get('translate', 0) > 0:
            images = batch_crop(self.proc_images['pad'], self.images.shape[-2])
        elif self.aug.get('flip', False):
            images = self.proc_images['flip']
        else:
            images = self.proc_images['norm']
        if self.aug.get('flip', False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1
        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

#############################################
#            Network Components             #
#############################################

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum, eps=1e-12, weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    """Standard ConvGroup (baseline)."""
    def __init__(self, channels_in, channels_out, batchnorm_momentum):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, batchnorm_momentum)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, batchnorm_momentum)
        
        if MODIFICATIONS['mish_activation']:
            self.activ = Mish()
        else:
            self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

class ConvGroupAntiAliased(nn.Module):
    """ConvGroup with anti-aliased pooling."""
    def __init__(self, channels_in, channels_out, batchnorm_momentum):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = MaxBlurPool2d(channels_out, filter_size=MODIFICATIONS['blur_filter_size'])
        self.norm1 = BatchNorm(channels_out, batchnorm_momentum)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, batchnorm_momentum)
        
        if MODIFICATIONS['mish_activation']:
            self.activ = Mish()
        else:
            self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

#############################################
#            Network Definition             #
#############################################

def make_net():
    widths = hyp['net']['widths']
    batchnorm_momentum = hyp['net']['batchnorm_momentum']
    whiten_kernel_size = 2
    whiten_width = 2 * 3 * whiten_kernel_size**2
    
    # Choose ConvGroup based on modification toggle
    if MODIFICATIONS['antialiased_pooling']:
        CG = ConvGroupAntiAliased
    else:
        CG = ConvGroup
    
    # Choose activation for first layer
    if MODIFICATIONS['mish_activation']:
        first_activ = Mish()
    else:
        first_activ = nn.GELU()
    
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        first_activ,
        CG(whiten_width,     widths['block1'], batchnorm_momentum),
        CG(widths['block1'], widths['block2'], batchnorm_momentum),
        CG(widths['block2'], widths['block3'], batchnorm_momentum),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths['block3'], 10, bias=False),
        Mul(hyp['net']['scaling_factor']),
    )
    net[0].weight.requires_grad = False
    net = net.half().cuda()
    net = net.to(memory_format=torch.channels_last)
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net

#############################################
#       Whitening Conv Initialization       #
#############################################

def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()

def get_whitening_parameters(patches):
    n,c,h,w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)

def init_whitening_conv(layer, train_set, eps=5e-4):
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

############################################
#                Lookahead                 #
############################################

class LookaheadState:
    def __init__(self, net):
        self.net_ema = {k: v.clone() for k, v in net.state_dict().items()}

    def update(self, net, decay):
        for ema_param, net_param in zip(self.net_ema.values(), net.state_dict().values()):
            if net_param.dtype in (torch.half, torch.float):
                ema_param.lerp_(net_param, 1-decay)
                net_param.copy_(ema_param)

############################################
#                 Logging                  #
############################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ''
    for col in columns_list:
        print_string += '|  %s  ' % col
    print_string += '|'
    if is_head:
        print('-'*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print('-'*len(print_string))

logging_columns_list = ['run   ', 'epoch', 'train_loss', 'train_acc', 'val_acc', 'tta_val_acc', 'total_time_seconds']
def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = '{:0.4f}'.format(var)
        else:
            assert var is None
            res = ''
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)

############################################
#               Evaluation                 #
############################################

def infer(model, loader, tta_level=0):
    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,)*4, 'reflect')
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [infer_mirror(inputs_translate, net)
                                 for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])

def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

############################################
#          BN Momentum Update              #
############################################

def update_bn_momentum(model, momentum):
    """Update momentum for all BatchNorm layers."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, BatchNorm)):
            module.momentum = 1 - momentum

############################################
#                Training                  #
############################################

def main(run, seed):
    """
    Run a single training experiment.
    
    Returns:
        dict with 'accuracy', 'tta_accuracy', 'total_time_seconds'
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    batch_size = hyp['opt']['batch_size']
    epochs = hyp['opt']['train_epochs']
    momentum = hyp['opt']['momentum']
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp['opt']['lr'] / kilostep_scale
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']

    # Label smoothing (potentially scheduled)
    current_smoothing = hyp['opt']['label_smoothing']
    loss_fn = nn.CrossEntropyLoss(label_smoothing=current_smoothing, reduction='none')
    
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    train_loader = CifarLoader('cifar10', train=True, batch_size=batch_size, aug=hyp['aug'])
    if run == 'warmup':
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)
    total_train_steps = ceil(len(train_loader) * epochs)

    model = make_net()
    current_steps = 0

    norm_biases = [p for k, p in model.named_parameters() if 'norm' in k and p.requires_grad]
    other_params = [p for k, p in model.named_parameters() if 'norm' not in k and p.requires_grad]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    
    # Choose optimizer based on modification toggle
    if MODIFICATIONS['gradient_centralization']:
        optimizer = SGD_GC(param_configs, momentum=momentum, nesterov=True)
    else:
        optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

    def get_lr(step):
        warmup_steps = int(total_train_steps * 0.23)
        warmdown_steps = total_train_steps - warmup_steps
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        else:
            frac = (step - warmup_steps) / warmdown_steps
            return 1.0 * (1 - frac) + 0.07 * frac
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    alpha_schedule = 0.95**5 * (torch.arange(total_train_steps+1) / total_train_steps)**3
    lookahead_state = LookaheadState(model)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    total_time_seconds = 0.0

    starter.record()
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model[0], train_images)
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    for epoch in range(ceil(epochs)):
        model[0].bias.requires_grad = (epoch < hyp['opt']['whiten_bias_epochs'])

        ####################
        #     Training     #
        ####################

        starter.record()
        model.train()
        
        for inputs, labels in train_loader:
            # Mixup augmentation (if enabled)
            if MODIFICATIONS['mixup']:
                inputs, labels_a, labels_b, lam = mixup_data(
                    inputs, labels, alpha=MODIFICATIONS['mixup_alpha']
                )
                outputs = model(inputs)
                loss = mixup_criterion(
                    lambda p, t: loss_fn(p, t).sum(), 
                    outputs, labels_a, labels_b, lam
                )
            else:
                outputs = model(inputs)
                loss = loss_fn(outputs, labels).sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            current_steps += 1

            # BN momentum scheduling (if enabled)
            if MODIFICATIONS['bn_momentum_schedule']:
                frac = current_steps / total_train_steps
                bn_momentum = (MODIFICATIONS['bn_momentum_start'] * (1 - frac) + 
                              MODIFICATIONS['bn_momentum_end'] * frac)
                update_bn_momentum(model, bn_momentum)

            if current_steps % 5 == 0:
                lookahead_state.update(model, decay=alpha_schedule[current_steps].item())

            if current_steps >= total_train_steps:
                if lookahead_state is not None:
                    lookahead_state.update(model, decay=1.0)
                break

        ender.record()
        torch.cuda.synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)

        ####################
        #    Evaluation    #
        ####################

        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        train_loss = loss.item() / batch_size
        val_acc = evaluate(model, test_loader, tta_level=0)
        print_training_details(locals(), is_final_entry=False)
        run = None

    ####################
    #  TTA Evaluation  #
    ####################

    starter.record()
    tta_val_acc = evaluate(model, test_loader, tta_level=hyp['net']['tta_level'])
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    epoch = 'eval'
    print_training_details(locals(), is_final_entry=True)

    return {
        'accuracy': val_acc,
        'tta_accuracy': tta_val_acc,
        'total_time_seconds': total_time_seconds,
    }


def get_experiment_name():
    """Generate experiment name from active modifications."""
    if ARGS.experiment_name:
        return ARGS.experiment_name
    
    active_mods = []
    if MODIFICATIONS['gradient_centralization']:
        active_mods.append('gc')
    if MODIFICATIONS['antialiased_pooling']:
        active_mods.append(f'aa{MODIFICATIONS["blur_filter_size"]}')
    if MODIFICATIONS['mish_activation']:
        active_mods.append('mish')
    if MODIFICATIONS['bn_momentum_schedule']:
        active_mods.append('bnm')
    if MODIFICATIONS['mixup']:
        active_mods.append(f'mix{MODIFICATIONS["mixup_alpha"]}')
    if MODIFICATIONS['label_smoothing_schedule']:
        active_mods.append('lss')
    
    if not active_mods:
        return 'baseline'
    return '_'.join(active_mods)


def get_git_commit():
    """Try to get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except:
        pass
    return 'unknown'


if __name__ == "__main__":
    experiment_name = get_experiment_name()
    timestamp = datetime.now().isoformat()
    
    # Print experiment info
    print("=" * 70)
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Timestamp: {timestamp}")
    print(f"Runs: {ARGS.n_runs}, Seed start: {ARGS.seed_start}")
    print("-" * 70)
    print("ACTIVE MODIFICATIONS:")
    active_count = 0
    if MODIFICATIONS['gradient_centralization']:
        print(f"  - Gradient Centralization")
        active_count += 1
    if MODIFICATIONS['antialiased_pooling']:
        print(f"  - Anti-Aliased Pooling (filter_size={MODIFICATIONS['blur_filter_size']})")
        active_count += 1
    if MODIFICATIONS['mish_activation']:
        print(f"  - Mish Activation")
        active_count += 1
    if MODIFICATIONS['bn_momentum_schedule']:
        print(f"  - BN Momentum Schedule ({MODIFICATIONS['bn_momentum_start']} -> {MODIFICATIONS['bn_momentum_end']})")
        active_count += 1
    if MODIFICATIONS['mixup']:
        print(f"  - Mixup (alpha={MODIFICATIONS['mixup_alpha']})")
        active_count += 1
    if MODIFICATIONS['label_smoothing_schedule']:
        print(f"  - Label Smoothing Schedule ({MODIFICATIONS['label_smoothing_start']} -> {MODIFICATIONS['label_smoothing_end']})")
        active_count += 1
    if active_count == 0:
        print("  (None - running baseline)")
    print("=" * 70)
    
    # Collect environment info
    env_info = {
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'git_commit': get_git_commit(),
    }
    print(f"GPU: {env_info['gpu']}")
    print(f"PyTorch: {env_info['torch_version']}, CUDA: {env_info['cuda_version']}")
    print("=" * 70)
    
    # Read source code for logging
    with open(sys.argv[0]) as f:
        code = f.read()

    # Run experiments
    print_columns(logging_columns_list, is_head=True)
    
    run_results = []
    for i in range(ARGS.n_runs):
        seed = ARGS.seed_start + i
        result = main(run=i, seed=seed)
        result['seed'] = seed
        result['run'] = i
        run_results.append(result)
    
    # Compute statistics
    accs = torch.tensor([r['tta_accuracy'] for r in run_results])
    times = torch.tensor([r['total_time_seconds'] for r in run_results])
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"TTA Accuracy:  Mean={accs.mean():.4f}  Std={accs.std():.4f}  "
          f"Min={accs.min():.4f}  Max={accs.max():.4f}")
    print(f"Time (sec):    Mean={times.mean():.4f}  Std={times.std():.4f}  "
          f"Min={times.min():.4f}  Max={times.max():.4f}")
    
    # 95% confidence intervals (using t-distribution)
    n = len(accs)
    if n > 1:
        import math
        t_value = 1.96 if n > 30 else 2.045  # Approximate for n~30
        acc_ci = t_value * accs.std() / math.sqrt(n)
        time_ci = t_value * times.std() / math.sqrt(n)
        print(f"\n95% CI (Accuracy): [{accs.mean() - acc_ci:.4f}, {accs.mean() + acc_ci:.4f}]")
        print(f"95% CI (Time):     [{times.mean() - time_ci:.4f}, {times.mean() + time_ci:.4f}]")
    print("=" * 70)
    
    # Save comprehensive log
    log = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'args': vars(ARGS),
        'modifications': MODIFICATIONS,
        'hyperparameters': hyp,
        'environment': env_info,
        'runs': run_results,
        'summary': {
            'n_runs': ARGS.n_runs,
            'seed_start': ARGS.seed_start,
            'accuracy_mean': float(accs.mean()),
            'accuracy_std': float(accs.std()),
            'accuracy_min': float(accs.min()),
            'accuracy_max': float(accs.max()),
            'time_mean': float(times.mean()),
            'time_std': float(times.std()),
            'time_min': float(times.min()),
            'time_max': float(times.max()),
        },
        'code': code,
        # Also save as tensors for easy torch.load
        'accs': accs,
        'times': times,
    }
    
    # Create output directory
    log_dir = os.path.join(ARGS.output_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Save as both .pt (torch) and .json (human readable)
    torch_log_path = os.path.join(log_dir, 'log.pt')
    json_log_path = os.path.join(log_dir, 'log.json')
    
    torch.save(log, torch_log_path)
    
    # JSON-safe version (no tensors)
    json_log = {k: v for k, v in log.items() if k not in ['code', 'accs', 'times']}
    json_log['accs'] = accs.tolist()
    json_log['times'] = times.tolist()
    with open(json_log_path, 'w') as f:
        json.dump(json_log, f, indent=2)
    
    print(f"\nLogs saved to:")
    print(f"  {os.path.abspath(torch_log_path)}")
    print(f"  {os.path.abspath(json_log_path)}")
