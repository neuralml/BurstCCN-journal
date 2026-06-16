from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def get_activation_function(activation_function):
    if activation_function == "sigmoid":
        return SigmoidActivation()
    elif activation_function == "relu":
        return ReLUActivation()
    elif activation_function == "softmax":
        return SoftmaxActivation()
    elif activation_function == "ln_relu":
        return LNReLUActivation()
    elif activation_function == "relu_ln":
        return ReLULNActivation()
    elif activation_function == "ms_relu_n":
        return MSReLUNActivation()
    elif activation_function == "ms_softplus_n":
        return MSSoftplusNActivation()
    elif activation_function == "ln_sigmoid":
        return LNSigmoidActivation()
    elif activation_function == "relu_rms":
        return ReLURMSNormActivation()
    else:
        raise ValueError(f"Unknown activation function {activation_function}")


class ActivationFunction:
    def forward(self, soma: torch.Tensor):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)


@dataclass
class SigmoidCtx:
    y: torch.Tensor


class SigmoidActivation(ActivationFunction):
    def forward(self, soma: torch.Tensor):
        y = torch.sigmoid(soma)
        return y, SigmoidCtx(y=y)

    def backward(self, delta: torch.Tensor, ctx: SigmoidCtx):
        y = ctx.y
        return delta * (y * (1 - y))


@dataclass
class LNSigmoidCtx:
    y: torch.Tensor
    xhat: torch.Tensor
    std: torch.Tensor
    gamma: float


class LNSigmoidActivation(ActivationFunction):
    def __init__(self, gamma: float = 1.0, beta: float = 0.0):
        self.gamma = gamma
        self.beta = beta

    def forward(self, soma: torch.Tensor):
        #TMP layer_norm
        eps = 1e-5
        x = soma

        mu = x.mean(dim=1, keepdim=True)
        xc = x - mu
        var = (xc * xc).mean(dim=1, keepdim=True)
        std = (var + eps).sqrt()
        xhat = xc / std
        u = self.gamma * xhat + self.beta
        # self.soma = xhat
        y = torch.sigmoid(u)

        return y, LNSigmoidCtx(y=y, std=std, xhat=xhat, gamma=self.gamma)

    def backward(self, delta: torch.Tensor, ctx: LNSigmoidCtx):
        y = ctx.y

        du = delta * (y * (1 - y))
        dgamma = (du * ctx.xhat).mean()
        dbeta = du.mean()

        g = du * ctx.gamma

        g_mean = g.mean(dim=1, keepdim=True)
        gxhat_mean = (g * ctx.xhat).mean(dim=1, keepdim=True)

        g_x = (g - g_mean - ctx.xhat * gxhat_mean) / ctx.std

        self.gamma -= 0.01 * dgamma
        self.beta -= 0.01 * dbeta
        return g_x


@dataclass
class ReLUCtx:
    soma: torch.Tensor
    mask: torch.Tensor  # boolean mask (soma > 0)


class ReLUActivation(ActivationFunction):
    def forward(self, soma: torch.Tensor):
        mask = soma > 0
        # y = soma.clamp_min(0.0)
        y = torch.relu(soma)

        return y, ReLUCtx(mask=mask, soma=soma)

    def backward(self, delta: torch.Tensor, ctx: ReLUCtx):
        # m = torch.clamp(ctx.soma / 0.1, min=0.0, max=1.0)
        m = torch.clamp(ctx.soma, min=0.0, max=1.0)

        return delta * ctx.mask * m


@dataclass
class SoftmaxCtx:
    y: torch.Tensor  # softmax output (optional to store)


class SoftmaxActivation(ActivationFunction):
    def __init__(self, dim: int = 1):
        self.dim = dim

    def forward(self, soma: torch.Tensor):
        y = torch.softmax(soma, dim=self.dim)
        return y, SoftmaxCtx(y=y)

    def backward(self, delta: torch.Tensor, ctx: SoftmaxCtx):
        # In your framework the loss already produced dL/dy,
        # so dz = delta.
        return delta


@dataclass
class LNReLUCtx:
    zhat: torch.Tensor
    inv_std: torch.Tensor
    relu_mask: torch.Tensor
    ln: torch.Tensor


class LNReLUActivation(ActivationFunction):
    def __init__(self, gamma_init: float = 1.5, beta_init: float = 0.0):
        self.gamma = gamma_init
        self.beta = beta_init

    def forward(self, z: torch.Tensor, eps: float = 1e-5):
        dims = tuple(range(1, z.dim()))

        # z: (N, D)
        mu = z.mean(dim=dims, keepdim=True)  # (N,1)
        var = ((z - mu) ** 2).mean(dim=dims, keepdim=True)  # (N,1)
        inv_std = (var + eps).rsqrt()  # (N,1)
        zhat = (z - mu) * inv_std  # (N,D)

        ln = zhat * self.gamma + self.beta  # (N,D)

        # y = torch.clamp_min(ln, 0.0)
        y = torch.relu(ln)
        relu_mask = (ln > 0)

        ctx = LNReLUCtx(zhat=zhat, inv_std=inv_std, relu_mask=relu_mask, ln=ln)
        return y, ctx

    def backward(self, dy: torch.Tensor, ctx: LNReLUCtx):
        dims = tuple(range(1, ctx.zhat.dim()))

        # dy: (N, D) gradient wrt output y
        zhat, inv_std, relu_mask = ctx.zhat, ctx.inv_std, ctx.relu_mask

        ramp_mask = torch.clamp(ctx.ln, min=0.0, max=1.0)

        # ReLU backward
        # dln = dy * relu_mask  # (N,D)
        dln = dy * relu_mask * ramp_mask

        # backprop through affine to zhat
        dzhat = dln * self.gamma  # (N,D)

        # LayerNorm backward (per sample over features)
        mean_dzhat = dzhat.mean(dim=dims, keepdim=True)  # (N,1)
        mean_dzhat_zhat = (dzhat * zhat).mean(dim=dims, keepdim=True)  # (N,1)
        dz = inv_std * (dzhat - mean_dzhat - zhat * mean_dzhat_zhat)  # (N,D)

        return dz


@dataclass
class ReLULNCtx:
    ahat: torch.Tensor
    inv_std: torch.Tensor
    relu_mask: torch.Tensor
    z: torch.Tensor


class ReLULNActivation(ActivationFunction):
    def __init__(self, gamma_init: float = 1.0, beta_init: float = 0.0):
        self.gamma = gamma_init
        self.beta = beta_init

    def forward(self, z: torch.Tensor, eps: float = 1e-5):
        # z: (N, D)
        dims = tuple(range(1, z.dim()))

        a = torch.relu(z)
        relu_mask = (z > 0)

        # Full
        mu = a.mean(dim=dims, keepdim=True)  # (N,1)
        var = ((a - mu) ** 2).mean(dim=dims, keepdim=True)  # (N,1)
        inv_std = (var + eps).rsqrt()  # (N,1)
        ahat = (a - mu) * inv_std  # (N,D)


        # # --- std-scale only ---
        # var = (a ** 2).mean(dim=dims, keepdim=True)  # (N,1)
        # inv_std = (var + eps).rsqrt()  # (N,1)
        # ahat = a * inv_std  # (N,D)

        y = ahat * self.gamma + self.beta

        ctx = ReLULNCtx(ahat=ahat, inv_std=inv_std, relu_mask=relu_mask, z=z)
        return y, ctx

    def backward(self, dy: torch.Tensor, ctx: ReLULNCtx):
        dims = tuple(range(1, ctx.ahat.dim()))

        ahat, inv_std, relu_mask = ctx.ahat, ctx.inv_std, ctx.relu_mask

        # backprop through affine
        dahat = dy * self.gamma  # (N,D)


        mean_dahat = dahat.mean(dim=dims, keepdim=True)
        mean_dahat_ahat = (dahat * ahat).mean(dim=dims, keepdim=True)


        # --- full
        da = inv_std * (dahat - mean_dahat - ahat * mean_dahat_ahat)
        # # --- std-scale only backward (matches your enabled code) ---
        # da = inv_std * (dahat - ahat * mean_dahat_ahat)  # (N,D)

        # ReLU backward to soma
        ramp_mask = torch.clamp(ctx.z, min=0.0, max=1.0)

        # dz = da * relu_mask
        dz = da * relu_mask * ramp_mask
        return dz







@dataclass
class MSReLUNCtx:
    ahat: torch.Tensor
    inv_std: torch.Tensor
    relu_mask: torch.Tensor
    zhat: torch.Tensor


class MSReLUNActivation(ActivationFunction):
    def __init__(self, gamma_init: float = 1.0, beta_init: float = 0.0):
        self.gamma = gamma_init
        self.beta = beta_init

    def forward(self, z: torch.Tensor, eps: float = 1e-5):
        dims = tuple(range(1, z.dim()))

        mu = z.mean(dim=dims, keepdim=True)
        zhat = z - mu


        a = torch.relu(zhat)
        relu_mask = (zhat > 0)

        # --- std-scale only ---
        var = (a ** 2).mean(dim=dims, keepdim=True)  # (N,1)
        inv_std = (var + eps).rsqrt()  # (N,1)
        ahat = a * inv_std  # (N,D)

        y = ahat * self.gamma + self.beta

        ctx = MSReLUNCtx(ahat=ahat, inv_std=inv_std, relu_mask=relu_mask, zhat=zhat)
        return y, ctx

    def backward(self, dy: torch.Tensor, ctx: MSReLUNCtx):
        dims = tuple(range(1, ctx.ahat.dim()))

        ahat, inv_std, relu_mask = ctx.ahat, ctx.inv_std, ctx.relu_mask

        # backprop through affine
        dahat = dy * self.gamma  # (N,D)

        # mean_dahat = dahat.mean(dim=dims, keepdim=True)
        mean_dahat_ahat = (dahat * ahat).mean(dim=dims, keepdim=True)

        da = inv_std * (dahat - ahat * mean_dahat_ahat)  # (N,D)

        # ReLU backward to soma
        ramp_mask = torch.clamp(ctx.zhat, min=0.0, max=1.0)

        dzhat = da * relu_mask
        # dzhat = da * ramp_mask

        mean_dzhat = dzhat.mean(dim=dims, keepdim=True)
        dz = dzhat - mean_dzhat

        return dz * ramp_mask


class MSSoftplusNCtx:
    def __init__(self, ahat, inv_std, zhat):
        self.ahat = ahat
        self.inv_std = inv_std
        self.zhat = zhat

class MSSoftplusNActivation(ActivationFunction):
    def __init__(self, gamma_init: float = 1.0, beta_init: float = 0.0, softplus_beta: float = 2.0, threshold: float = 10.0):
        self.gamma = gamma_init
        self.beta = beta_init
        self.softplus_beta = softplus_beta
        self.threshold = threshold

    def forward(self, z: torch.Tensor, eps: float = 1e-5):
        dims = tuple(range(1, z.dim()))

        mu = z.mean(dim=dims, keepdim=True)
        zhat = z - mu

        # --- Softplus instead of ReLU ---
        a = F.softplus(zhat, beta=self.softplus_beta, threshold=self.threshold)

        # --- std-scale only (unchanged) ---
        var = (a ** 2).mean(dim=dims, keepdim=True)
        inv_std = (var + eps).rsqrt()
        ahat = a * inv_std

        y = ahat * self.gamma + self.beta
        ctx = MSSoftplusNCtx(ahat=ahat, inv_std=inv_std, zhat=zhat)
        return y, ctx

    def backward(self, dy: torch.Tensor, ctx: MSSoftplusNCtx):
        dims = tuple(range(1, ctx.ahat.dim()))

        ahat, inv_std, zhat = ctx.ahat, ctx.inv_std, ctx.zhat

        # backprop through affine
        dahat = dy * self.gamma

        mean_dahat_ahat = (dahat * ahat).mean(dim=dims, keepdim=True)

        # backprop through normalization (unchanged)
        da = inv_std * (dahat - ahat * mean_dahat_ahat)

        # --- Softplus backward: multiply by sigmoid(zhat) ---
        dsoft = torch.sigmoid(self.softplus_beta * zhat)  # derivative wrt zhat when using beta
        dzhat = da * dsoft

        mean_dzhat = dzhat.mean(dim=dims, keepdim=True)
        dz = dzhat - mean_dzhat
        return dz
















def ln_relu_forward(self, z, eps=1e-5):
    mu = z.mean(dim=1, keepdim=True)  # (N, 1)
    var = ((z - mu) ** 2).mean(dim=1, keepdim=True)  # (N, 1)
    inv_std = (var + eps).rsqrt()  # (N, 1)
    zhat = (z - mu) * inv_std  # (N, Dout)

    # Mean-shift only
    # zhat = (z - mu)  # (N, Dout)

    # std-scale only
    # var = ((z ** 2).mean(dim=1, keepdim=True))  # (N,1)
    # inv_std = (var + eps).rsqrt()  # (N, 1)
    # zhat = z * inv_std  # (N, Dout)

    # affine
    ln = zhat * self.gamma.unsqueeze(0) + self.beta.unsqueeze(0)  # (N, Dout)

    # ReLU
    y = torch.clamp_min(ln, 0.0)
    relu_mask = (ln > 0)

    self.f_cache = (zhat, inv_std, relu_mask)
    return y


def ln_relu_backward(self, dy):
    """
    dy: (N, Dout) gradient wrt output y
    returns: dx, dW, db, dgamma, dbeta
    """
    zhat, inv_std, relu_mask = self.f_cache
    N, Dout = dy.shape

    # ReLU backward
    dln = dy * relu_mask  # (N, Dout)

    # affine backward
    dbeta = dln.mean(dim=0)  # (Dout,)
    dgamma = (dln * zhat).mean(dim=0)  # (Dout,)
    dzhat = dln * self.gamma.unsqueeze(0)  # (N, Dout)

    # LayerNorm backward (per sample over features)
    # dz = inv_std * (dzhat - mean(dzhat) - zhat * mean(dzhat * zhat))
    mean_dzhat = dzhat.mean(dim=1, keepdim=True)  # (N, 1)
    mean_dzhat_zhat = (dzhat * zhat).mean(dim=1, keepdim=True)  # (N, 1)
    dz = inv_std * (dzhat - mean_dzhat - zhat * mean_dzhat_zhat)  # (N, Dout)

    # Mean-shift only
    # dz = dzhat - mean_dzhat  # (N, Dout)

    # std-scale only
    # dz = inv_std * (dzhat - zhat * mean_dzhat_zhat)

    # self.beta -= 0.0001 * dbeta
    # self.gamma -= 0.0001 * dgamma

    return dz

def relu_ln_forward(self, z, eps=1e-5):
    # z: (N, D)
    a = torch.relu(z)
    relu_mask = (z > 0)

    mu = a.mean(dim=1, keepdim=True)  # (N,1)
    var = ((a - mu) ** 2).mean(dim=1, keepdim=True)  # (N,1)
    inv_std = (var + eps).rsqrt()  # (N,1)
    ahat = (a - mu) * inv_std  # (N,D)

    y = ahat * self.gamma.unsqueeze(0) + self.beta.unsqueeze(0)

    self.f_cache = (ahat, inv_std, relu_mask)
    return y

def relu_ln_backward(self, dy):
    # dy: (N, D) gradient wrt LN output
    ahat, inv_std, relu_mask = self.f_cache

    # affine grads
    dbeta = dy.mean(dim=0)
    dgamma = (dy * ahat).mean(dim=0)
    # self.beta.grad = dbeta
    # self.gamma.grad = dgamma

    # backprop through affine
    dahat = dy * self.gamma.unsqueeze(0)

    # LN backward on a
    mean_dahat = dahat.mean(dim=1, keepdim=True)
    mean_dahat_ahat = (dahat * ahat).mean(dim=1, keepdim=True)
    da = inv_std * (dahat - mean_dahat - ahat * mean_dahat_ahat)

    # ReLU backward to soma
    dz = da * relu_mask
    return dz


@dataclass
class ReLURMSNormCtx:
    ahat: torch.Tensor
    inv_rms: torch.Tensor
    a: torch.Tensor  # store activity for gating


class ReLURMSNormActivation(ActivationFunction):
    def __init__(self, gamma_init: float = 1.0, beta_init: float = 0.0):
        self.gamma = gamma_init
        self.beta = beta_init

    def forward(self, z: torch.Tensor, eps: float = 1e-5):
        dims = tuple(range(1, z.dim()))

        a = torch.relu(z)

        rms2 = (a * a).mean(dim=dims, keepdim=True)
        inv_rms = (rms2 + eps).rsqrt()

        ahat = a * inv_rms
        y = ahat * self.gamma + self.beta

        ctx = ReLURMSNormCtx(ahat=ahat, inv_rms=inv_rms, a=a)
        return y, ctx

    def _backward_impl(self, dy: torch.Tensor, ctx: ReLURMSNormCtx, use_surrogate: bool):
        dims = tuple(range(1, ctx.ahat.dim()))
        ahat, inv_rms, a = ctx.ahat, ctx.inv_rms, ctx.a

        dahat = dy * self.gamma
        mean_dahat_ahat = (dahat * ahat).mean(dim=dims, keepdim=True)

        da = inv_rms * (dahat - ahat * mean_dahat_ahat)

        # activity-bounded surrogate ReLU derivative
        # gate = torch.clamp(ahat * 5, 0.0, 1.0)   # same as clamp(z,0,1) after ReLU
        if use_surrogate:
            # activity-bounded surrogate ReLU derivative
            gate = torch.clamp(ahat * 5, 0.0, 1.0)
        else:
            # true ReLU derivative
            gate = (a > 0).to(a.dtype)

        dz = da * gate

        return dz

    def backward(self, dy: torch.Tensor, ctx: ReLURMSNormCtx):
        return self._backward_impl(dy, ctx, use_surrogate=False)

    def backward_surrogate(self, dy: torch.Tensor, ctx: ReLURMSNormCtx):
        return self._backward_impl(dy, ctx, use_surrogate=True)
