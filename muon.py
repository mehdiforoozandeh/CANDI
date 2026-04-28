import torch
import torch.distributed as dist

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power of a matrix G
    (i.e., the orthogonal matrix UV^T where G = USV^T).
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.775, 2.0315)
    X = G.bfloat16()
    
    # Ensure spectral norm is <= 1
    X /= (X.norm() + eps) 
    
    if G.size(0) > G.size(1):
        X = X.T
        
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    if G.size(0) > G.size(1):
        X = X.T
        
    return X.type_as(G)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    
    Muon internally uses AdamW for 1D tensors (biases, layernorm gains) 
    and embeddings, and the pure Muon update for 2D tensors (weights).
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, adamw_params=None):
        
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        
        params = list(params)
        
        # Separate parameters into those for Muon and those for AdamW
        muon_params = []
        adamw_params_list = []
        
        if adamw_params is None:
            adamw_params = {}
            
        # Default AdamW params if not provided
        adamw_lr = adamw_params.get('lr', 1e-3) # Typically lower than Muon lr
        adamw_betas = adamw_params.get('betas', (0.9, 0.999))
        adamw_wd = adamw_params.get('weight_decay', 0.01)
        
        # Simple heuristic: 2D -> Muon, <2D -> AdamW
        # Also typically embeddings are AdamW
        for p in params:
            if p.ndim == 2:
                muon_params.append(p)
            else:
                adamw_params_list.append(p)
                
        super().__init__([
            {'params': muon_params, 'optimizer': 'muon'},
            {'params': adamw_params_list, 'optimizer': 'adamw', 'lr': adamw_lr, 'betas': adamw_betas, 'weight_decay': adamw_wd}
        ], defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group['optimizer'] == 'muon':
                self.step_muon(group)
            elif group['optimizer'] == 'adamw':
                self.step_adamw(group)
                
        return loss

    def step_muon(self, group):
        lr = group['lr']
        momentum = group['momentum']
        nesterov = group['nesterov']
        ns_steps = group['ns_steps']
        
        for p in group['params']:
            if p.grad is None:
                continue
                
            grad = p.grad
            state = self.state[p]
            
            # Init state
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = torch.zeros_like(p)
                
            buf = state['momentum_buffer']
            
            # Update momentum buffer
            buf.mul_(momentum).add_(grad)
            
            if nesterov:
                g = grad.add(buf, alpha=momentum)
            else:
                g = buf
            
            # Orthogonalize update
            if g.ndim == 2:
                g_orth = zeropower_via_newtonschulz5(g, steps=ns_steps)
            else:
                g_orth = g / (g.norm() + 1e-7)
                
            # Update param
            # Muon update: p -= lr * g_orth * (max(1, p.size(0)/p.size(1))**0.5) 
            scale = max(1, p.size(0)/p.size(1))**0.5
            p.data.add_(g_orth, alpha=-lr * scale)
            
    def step_adamw(self, group):
        lr = group['lr']
        beta1, beta2 = group['betas']
        weight_decay = group['weight_decay']
        eps = 1e-8
        
        for p in group['params']:
            if p.grad is None:
                continue
            
            grad = p.grad
            state = self.state[p]
            
            if 'step' not in state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            state['step'] += 1
            
            # Weight decay
            if weight_decay != 0:
                p.data.mul_(1 - lr * weight_decay)
            
            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            denom = exp_avg_sq.sqrt().add_(eps)
            
            step_size = lr * (1 - beta2 ** state['step'])**0.5 / (1 - beta1 ** state['step'])
            
            p.data.addcdiv_(exp_avg, denom, value=-step_size)




