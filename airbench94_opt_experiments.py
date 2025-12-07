# Based on legacy airbench94.py (uncompiled) from cifar10-airbench
# Adds optimizer toggles + time-to-94-or-10s early stop + experiment logging harness.

#############################################
#            Setup/Hyperparameters          #
#############################################

import os
import sys
import json
import uuid
import argparse
import subprocess
import datetime
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True

# Baseline hyperparams (unchanged)
hyp = {
    'opt': {
        'train_epochs': 9.9,    # now treated as a soft cap, since we early-stop
        'batch_size': 1024,
        'lr': 11.5,             # learning rate per 1024 examples
        'momentum': 0.85,
        'weight_decay': 0.0153, # weight decay per 1024 examples (decoupled)
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
#                 CLI                       #
#############################################

def build_argparser():
    p = argparse.ArgumentParser("Airbench94 CIFAR-10 Optimizer Experiments")

    # experiment
    p.add_argument("--n-runs", type=int, default=25)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="logs")
    p.add_argument("--data-dir", type=str, default="cifar10")

    # stopping conditions
    p.add_argument("--target-acc", type=float, default=0.94,
                   help="Stop training once val_acc >= target-acc")
    p.add_argument("--max-time-seconds", type=float, default=10.0,
                   help="Stop training once total_time_seconds exceeds this")

    # optimizer choice
    p.add_argument("--optimizer", type=str, default="sgd",
                   choices=["sgd", "lion", "sophia", "ademamix"],
                   help="Choose optimizer for ablation")

    # optional optimizer hyper overrides
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--momentum", type=float, default=None)  # used for sgd-like paths
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--train-epochs", type=float, default=None)
    p.add_argument("--label-smoothing", type=float, default=None)

    # Lion params
    p.add_argument("--lion-beta1", type=float, default=0.9)
    p.add_argument("--lion-beta2", type=float, default=0.99)

    # Sophia-like params (approx)
    p.add_argument("--sophia-beta1", type=float, default=0.965)
    p.add_argument("--sophia-beta2", type=float, default=0.99)
    p.add_argument("--sophia-rho", type=float, default=0.04)
    p.add_argument("--sophia-eps", type=float, default=1e-8)

    # AdEMAMix-like params (approx)
    p.add_argument("--ademamix-beta1", type=float, default=0.9)
    p.add_argument("--ademamix-beta2", type=float, default=0.999)
    p.add_argument("--ademamix-beta3", type=float, default=0.9999)
    p.add_argument("--ademamix-alpha", type=float, default=0.5)
    p.add_argument("--ademamix-eps", type=float, default=1e-8)

    return p

ARGS = build_argparser().parse_args()

# Apply CLI hyper overrides
if ARGS.lr is not None:
    hyp['opt']['lr'] = ARGS.lr
if ARGS.weight_decay is not None:
    hyp['opt']['weight_decay'] = ARGS.weight_decay
if ARGS.momentum is not None:
    hyp['opt']['momentum'] = ARGS.momentum
if ARGS.batch_size is not None:
    hyp['opt']['batch_size'] = ARGS.batch_size
if ARGS.train_epochs is not None:
    hyp['opt']['train_epochs'] = ARGS.train_epochs
if ARGS.label_smoothing is not None:
    hyp['opt']['label_smoothing'] = ARGS.label_smoothing


#############################################
#            Utilities                      #
#############################################

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

def get_experiment_name():
    return f"cifar10_{ARGS.optimizer}"

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
        for k in self.aug.keys():
            assert k in ['flip', 'translate'], f'Unrecognized key: {k}'

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
    def __init__(self, channels_in, channels_out, batchnorm_momentum):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, batchnorm_momentum)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, batchnorm_momentum)
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
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width,     widths['block1'], batchnorm_momentum),
        ConvGroup(widths['block1'], widths['block2'], batchnorm_momentum),
        ConvGroup(widths['block2'], widths['block3'], batchnorm_momentum),
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
        print_string += '| %s ' % col
    print_string += '|'
    if is_head:
        print('-'*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print('-'*len(print_string))

logging_columns_list = ['run ', 'epoch', 'train_loss', 'train_acc', 'val_acc', 'tta_val_acc', 'total_time_seconds']

def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if isinstance(var, (int, str)):
            res = str(var)
        elif isinstance(var, float):
            res = '{:0.4f}'.format(var)
        else:
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
#        Optimizers (custom/light)         #
############################################

class Lion(torch.optim.Optimizer):
    """
    Standard Lion-style update:
      m <- beta1*m + (1-beta1)*g
      p <- p * (1 - lr*wd) - lr * sign(m)
      m <- beta2*m + (1-beta2)*g  (often applied as a second EMA)
    This implementation uses decoupled weight decay.
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m = state["m"]
                m.mul_(beta1).add_(g, alpha=1 - beta1)

                if wd != 0:
                    p.mul_(1 - lr * wd)

                p.add_(m.sign(), alpha=-lr)

                # second EMA update (as used in many reference impls)
                m.mul_(beta2).add_(g, alpha=1 - beta2)

        return loss


class SophiaLike(torch.optim.Optimizer):
    """
    Lightweight 'Sophia-like' diagonal second-order approximation.
    This is NOT a faithful reproduction of the official algorithm,
    but a reasonable class-project ablation.

    Maintains:
      m <- beta1*m + (1-beta1)*g
      h <- beta2*h + (1-beta2)*(g*g)   # proxy for diag curvature
    Update:
      u <- clamp(m / (h + eps), -rho, rho)
      p <- p*(1 - lr*wd) - lr*u
    """
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04, eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, rho=rho, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["h"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m = state["m"]
                h = state["h"]

                m.mul_(beta1).add_(g, alpha=1 - beta1)
                h.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                if wd != 0:
                    p.mul_(1 - lr * wd)

                u = m / (h.add(eps))
                u.clamp_(-rho, rho)
                p.add_(u, alpha=-lr)

        return loss


class AdEMAMixLike(torch.optim.Optimizer):
    """
    Lightweight 'AdEMAMix-like' optimizer for ablation.
    Keeps two EMAs of grad with different decay and mixes them.

    m1 <- beta1*m1 + (1-beta1)*g
    m2 <- beta3*m2 + (1-beta3)*g
    v  <- beta2*v  + (1-beta2)*g^2
    m  <- alpha*m1 + (1-alpha)*m2
    p  <- p*(1 - lr*wd) - lr * m / (sqrt(v)+eps)
    """
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.999, beta3=0.9999,
                 alpha=0.5, eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, beta3=beta3,
                        alpha=alpha, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            b3 = group["beta3"]
            alpha = group["alpha"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["m1"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["m2"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"]  = torch.zeros_like(p, memory_format=torch.preserve_format)

                m1 = state["m1"]
                m2 = state["m2"]
                v  = state["v"]

                m1.mul_(b1).add_(g, alpha=1 - b1)
                m2.mul_(b3).add_(g, alpha=1 - b3)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)

                if wd != 0:
                    p.mul_(1 - lr * wd)

                m = alpha * m1 + (1 - alpha) * m2
                denom = v.sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-lr)

        return loss


def build_optimizer(model, lr, wd, lr_biases, wd_biases):
    """
    Keep the same parameter grouping logic as baseline:
    - BatchNorm-related parameters get lr_biases
    - Others get lr
    Weight decay is decoupled in custom optimizers.
    """
    norm_biases = [p for k, p in model.named_parameters() if 'norm' in k and p.requires_grad]
    other_params = [p for k, p in model.named_parameters() if 'norm' not in k and p.requires_grad]

    opt_name = ARGS.optimizer

    if opt_name == "sgd":
        param_configs = [
            dict(params=norm_biases, lr=lr_biases, weight_decay=wd_biases),
            dict(params=other_params, lr=lr, weight_decay=wd),
        ]
        return torch.optim.SGD(param_configs, momentum=hyp['opt']['momentum'], nesterov=True)

    elif opt_name == "lion":
        beta1, beta2 = ARGS.lion_beta1, ARGS.lion_beta2
        # Provide two param groups to preserve bias_scaler behavior
        param_configs = [
            dict(params=norm_biases, lr=lr_biases, weight_decay=wd_biases, betas=(beta1, beta2)),
            dict(params=other_params, lr=lr, weight_decay=wd, betas=(beta1, beta2)),
        ]
        return Lion(param_configs)

    elif opt_name == "sophia":
        b1, b2 = ARGS.sophia_beta1, ARGS.sophia_beta2
        rho, eps = ARGS.sophia_rho, ARGS.sophia_eps
        param_configs = [
            dict(params=norm_biases, lr=lr_biases, weight_decay=wd_biases,
                 betas=(b1, b2), rho=rho, eps=eps),
            dict(params=other_params, lr=lr, weight_decay=wd,
                 betas=(b1, b2), rho=rho, eps=eps),
        ]
        return SophiaLike(param_configs)

    elif opt_name == "ademamix":
        param_configs = [
            dict(params=norm_biases, lr=lr_biases, weight_decay=wd_biases,
                 beta1=ARGS.ademamix_beta1, beta2=ARGS.ademamix_beta2,
                 beta3=ARGS.ademamix_beta3, alpha=ARGS.ademamix_alpha,
                 eps=ARGS.ademamix_eps),
            dict(params=other_params, lr=lr, weight_decay=wd,
                 beta1=ARGS.ademamix_beta1, beta2=ARGS.ademamix_beta2,
                 beta3=ARGS.ademamix_beta3, alpha=ARGS.ademamix_alpha,
                 eps=ARGS.ademamix_eps),
        ]
        return AdEMAMixLike(param_configs)

    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


############################################
#                Training                  #
############################################

def main(run, seed=0):
    set_seed(seed)

    batch_size = hyp['opt']['batch_size']
    epochs_cap = hyp['opt']['train_epochs']

    # For baseline SGD decoupling math:
    momentum = hyp['opt']['momentum']
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))

    # Base per-1024 lr & wd converted to actual update scale
    # We'll reuse these conversions for all optimizers for consistency,
    # but expect retuning for non-SGD methods.
    lr = hyp['opt']['lr'] / kilostep_scale
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']

    # For SGD weight-decay parameterization, baseline used wd/lr trick.
    # For decoupled optimizers (Lion/Sophia/AdEMAMix), we pass a plain decoupled wd.
    # We'll define both:
    wd_biases_decoupled = wd  # simple choice
    wd_other_decoupled = wd

    # For SGD's coupled parameter config in original code:
    # weight_decay set to (wd / lr_group)
    wd_biases_sgd = wd / lr_biases
    wd_other_sgd = wd / lr

    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')

    test_loader = CifarLoader(ARGS.data_dir, train=False, batch_size=2000)
    train_loader = CifarLoader(ARGS.data_dir, train=True, batch_size=batch_size, aug=hyp['aug'])

    total_train_steps_cap = ceil(len(train_loader) * epochs_cap)

    model = make_net()
    current_steps = 0

    optimizer = build_optimizer(
        model,
        lr=lr,
        wd=wd_other_decoupled if ARGS.optimizer != "sgd" else wd_other_sgd,
        lr_biases=lr_biases,
        wd_biases=wd_biases_decoupled if ARGS.optimizer != "sgd" else wd_biases_sgd,
    )

    # Keep the original LR schedule ONLY for SGD.
    # For the newer optimizers, you can try both with/without schedule later;
    # this keeps the ablation clean and avoids pretending schedule-free is implemented here.
    if ARGS.optimizer == "sgd":
        def get_lr(step):
            warmup_steps = int(total_train_steps_cap * 0.23)
            warmdown_steps = total_train_steps_cap - warmup_steps
            if step < warmup_steps:
                frac = step / warmup_steps
                return 0.2 * (1 - frac) + 1.0 * frac
            else:
                frac = (step - warmup_steps) / warmdown_steps
                return 1.0 * (1 - frac) + 0.07 * frac
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    else:
        scheduler = None

    alpha_schedule = 0.95**5 * (torch.arange(total_train_steps_cap+1) / total_train_steps_cap)**3
    lookahead_state = LookaheadState(model)

    # Timing
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    total_time_seconds = 0.0

    # Whitening init
    starter.record()
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model[0], train_images)
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    # Training until target acc or time cap
    reached_target = False
    stop_reason = "epochs_cap"

    for epoch in range(ceil(epochs_cap)):
        model[0].bias.requires_grad = (epoch < hyp['opt']['whiten_bias_epochs'])

        ####################
        # Training
        ####################
        starter.record()
        model.train()

        last_outputs, last_labels, last_loss = None, None, None

        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels).sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            current_steps += 1

            if current_steps % 5 == 0 and current_steps < len(alpha_schedule):
                lookahead_state.update(model, decay=alpha_schedule[current_steps].item())

            last_outputs, last_labels, last_loss = outputs, labels, loss

            # Step cap safety
            if current_steps >= total_train_steps_cap:
                if lookahead_state is not None:
                    lookahead_state.update(model, decay=1.0)
                break

        ender.record()
        torch.cuda.synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)

        # Compute train stats from last batch
        if last_outputs is not None:
            train_acc = (last_outputs.detach().argmax(1) == last_labels).float().mean().item()
            train_loss = last_loss.item() / batch_size
        else:
            train_acc, train_loss = 0.0, 0.0

        # Validation check
        val_acc = evaluate(model, test_loader, tta_level=0)

        print_training_details({
            "run": run,
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "tta_val_acc": None,
            "total_time_seconds": float(total_time_seconds),
        }, is_final_entry=False)

        # Early stop checks
        if val_acc >= ARGS.target_acc:
            reached_target = True
            stop_reason = "target_acc"
            break

        if total_time_seconds >= ARGS.max_time_seconds:
            stop_reason = "time_cap"
            break

    # Final TTA eval (timed) only once
    starter.record()
    tta_val_acc = evaluate(model, test_loader, tta_level=hyp['net']['tta_level'])
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    print_training_details({
        "run": "",
        "epoch": "eval",
        "train_loss": None,
        "train_acc": None,
        "val_acc": None,
        "tta_val_acc": float(tta_val_acc),
        "total_time_seconds": float(total_time_seconds),
    }, is_final_entry=True)

    return {
        "tta_accuracy": float(tta_val_acc),
        "val_accuracy": float(val_acc),
        "total_time_seconds": float(total_time_seconds),
        "stop_reason": stop_reason,
        "reached_target": bool(reached_target),
        "optimizer": ARGS.optimizer,
    }


#############################################
#                __main__                   #
#############################################

if __name__ == "__main__":
    experiment_name = get_experiment_name()
    timestamp = datetime.datetime.now().isoformat()

    print("=" * 70)
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Timestamp: {timestamp}")
    print(f"Runs: {ARGS.n_runs}, Seed start: {ARGS.seed_start}")
    print(f"Optimizer: {ARGS.optimizer}")
    print(f"Target acc: {ARGS.target_acc}, Max time: {ARGS.max_time_seconds}s")
    print("=" * 70)

    env_info = {
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'git_commit': get_git_commit(),
    }
    print(f"GPU: {env_info['gpu']}")
    print(f"PyTorch: {env_info['torch_version']}, CUDA: {env_info['cuda_version']}")
    print("=" * 70)

    with open(sys.argv[0]) as f:
        code = f.read()

    print_columns(logging_columns_list, is_head=True)

    run_results = []
    for i in range(ARGS.n_runs):
        seed = ARGS.seed_start + i
        result = main(run=i, seed=seed)
        result["seed"] = seed
        result["run"] = i
        run_results.append(result)

    # Stats
    accs = torch.tensor([r['tta_accuracy'] for r in run_results], dtype=torch.float32)
    times = torch.tensor([r['total_time_seconds'] for r in run_results], dtype=torch.float32)
    reached = sum(1 for r in run_results if r["reached_target"])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Reached target in {reached}/{ARGS.n_runs} runs")
    print(
        f"TTA Accuracy: Mean={accs.mean():.4f} Std={accs.std():.4f} "
        f"Min={accs.min():.4f} Max={accs.max():.4f}"
    )
    print(
        f"Time (sec): Mean={times.mean():.4f} Std={times.std():.4f} "
        f"Min={times.min():.4f} Max={times.max():.4f}"
    )
    print("=" * 70)

    log = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'args': vars(ARGS),
        'hyperparameters': hyp,
        'environment': env_info,
        'runs': run_results,
        'summary': {
            'n_runs': ARGS.n_runs,
            'seed_start': ARGS.seed_start,
            'reached_target_count': reached,
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
        'accs': accs,
        'times': times,
    }

    log_dir = os.path.join(
        ARGS.output_dir,
        f"{experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(log_dir, exist_ok=True)

    torch_log_path = os.path.join(log_dir, "log.pt")
    json_log_path = os.path.join(log_dir, "log.json")

    torch.save(log, torch_log_path)

    json_log = {k: v for k, v in log.items() if k not in ["code", "accs", "times"]}
    json_log["accs"] = accs.tolist()
    json_log["times"] = times.tolist()

    with open(json_log_path, "w") as f:
        json.dump(json_log, f, indent=2)

    print(f"\nLogs saved to:")
    print(f"  {os.path.abspath(torch_log_path)}")
    print(f"  {os.path.abspath(json_log_path)}")
