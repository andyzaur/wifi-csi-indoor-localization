"""PyTorch (MPS) shared-trunk localizer: joint (x,y) regression + cell classification.

Built for the small-data regime per the 2026-06-03 design synthesis:
- shared trunk + two heads (regularizes both tasks, halves params vs two nets)
- dropout / weight-decay / input Gaussian-noise / feature-dropout regularization
- minibatch SGD over GPU-resident tensors (NO DataLoader — at 25k x 195 the
  host<->device copies would dominate; the whole dataset lives on the GPU)
- early stopping on an inner-validation regression-median (the PRIMARY metric)
- float32 only (MPS has no float64)

Architecture/hyperparameters are chosen by Optuna on an inner-validation split in
train_v3.py — never on the test fold. This module is just the estimator.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
NORMS = {"batch": nn.BatchNorm1d, "layer": nn.LayerNorm, "none": nn.Identity}


def best_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class _Trunk(nn.Module):
    def __init__(self, d_in, width, depth, act, dropout, norm, residual):
        super().__init__()
        Norm, A = NORMS[norm], ACTS[act]
        self.inp = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList([
            nn.Sequential(Norm(width), A(), nn.Dropout(dropout), nn.Linear(width, width))
            for _ in range(depth)])
        self.residual = residual
        self.out = nn.Sequential(Norm(width), A(), nn.Dropout(dropout))

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h) if self.residual else b(h)
        return self.out(h)


class _Net(nn.Module):
    def __init__(self, d_in, n_cls, width, depth, act, dropout, norm, residual):
        super().__init__()
        self.trunk = _Trunk(d_in, width, depth, act, dropout, norm, residual)
        self.cls = nn.Linear(width, n_cls)
        self.reg = nn.Linear(width, 2)

    def forward(self, x):
        h = self.trunk(x)
        return self.cls(h), self.reg(h)


class TorchLocalizer:
    """Shared-trunk MLP estimator. Expects ALREADY-SCALED float32 X (StandardScaler
    fit on the train fold only). y_xy is standardized internally (inverted at
    predict); y_cls are integer class ids in [0, n_cls).

    Optional per-board train-time augmentation (default OFF — zero behavior
    change when off): `board_slices` lists one integer index array per RX board
    (that board's columns in X). Per minibatch sample, `aug_board_gain_std`
    multiplies each board's slice by an independent gain g ~ N(1, std) —
    mimicking the inter-session per-board gain/level shifts that dominate
    cross-session drift; `aug_board_drop_p` zeroes ONE uniformly-chosen board's
    slice with that probability — robustness to a dead/missing RX."""

    def __init__(self, n_cls, width=256, depth=2, act="gelu", dropout=0.2,
                 norm="batch", residual=False, lr=1e-3, weight_decay=1e-4,
                 batch_size=2048, max_epochs=300, patience=30, warmup_frac=0.05,
                 noise_std=0.0, feat_drop=0.0, label_smoothing=0.0, w_cls=0.5,
                 grad_clip=5.0, seed=0, device=None, verbose=False,
                 aug_board_gain_std=0.0, aug_board_drop_p=0.0, board_slices=None):
        self.n_cls = n_cls
        for k, v in dict(width=width, depth=depth, act=act, dropout=dropout,
                         norm=norm, residual=residual, lr=lr, weight_decay=weight_decay,
                         batch_size=batch_size, max_epochs=max_epochs, patience=patience,
                         warmup_frac=warmup_frac, noise_std=noise_std, feat_drop=feat_drop,
                         label_smoothing=label_smoothing, w_cls=w_cls, grad_clip=grad_clip,
                         seed=seed, verbose=verbose,
                         aug_board_gain_std=aug_board_gain_std,
                         aug_board_drop_p=aug_board_drop_p,
                         board_slices=board_slices).items():
            setattr(self, k, v)
        self.device = torch.device(device) if device else best_device()
        self.net_ = None
        self.val_history_ = []

    def _t(self, a, dtype=torch.float32):
        return torch.as_tensor(np.asarray(a, dtype=np.float32 if dtype == torch.float32 else None),
                               dtype=dtype, device=self.device)

    def fit(self, X, y_xy, y_cls, X_val=None, y_val_xy=None, y_val_cls=None,
            report_cb=None, init_state=None, y_stats=None):
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y_xy = np.asarray(y_xy, dtype=np.float32)
        # y_stats lets a fine-tune inherit the base model's room-scale (x,y)
        # standardization instead of recomputing it from a tiny target slice
        # (which would only span a few cells and squash predictions).
        if y_stats is not None:
            self.ymean_, self.ystd_ = np.asarray(y_stats[0]), np.asarray(y_stats[1])
        else:
            self.ymean_ = y_xy.mean(0)
            self.ystd_ = y_xy.std(0) + 1e-8
        Xt = self._t(X)
        yz = self._t((y_xy - self.ymean_) / self.ystd_)
        yc = torch.as_tensor(np.asarray(y_cls), dtype=torch.long, device=self.device)
        have_val = X_val is not None and y_val_xy is not None
        if have_val:
            Xv = self._t(np.asarray(X_val, dtype=np.float32))
            yv_xy = np.asarray(y_val_xy, dtype=np.float32)

        self.net_ = _Net(X.shape[1], self.n_cls, self.width, self.depth, self.act,
                         self.dropout, self.norm, self.residual).to(self.device)
        # warm-start (transfer/fine-tune): load pretrained weights before training
        if init_state is not None:
            self.net_.load_state_dict({k: v.to(self.device) for k, v in init_state.items()})
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        n = len(X)
        steps_per_epoch = max(1, n // self.batch_size)
        total_steps = steps_per_epoch * self.max_epochs
        warmup = max(1, int(total_steps * self.warmup_frac))

        def lr_at(step):  # linear warmup then cosine
            if step < warmup:
                return step / warmup
            prog = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        best_val, best_state, bad = np.inf, None, 0
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)

        # per-board augmentation masks: one row per board, 1.0 on its columns
        # (only built when the augmentation is on — defaults leave RNG untouched)
        board_masks = None
        if self.board_slices and (self.aug_board_gain_std or self.aug_board_drop_p):
            board_masks = torch.zeros((len(self.board_slices), X.shape[1]),
                                      dtype=torch.float32, device=self.device)
            for b, sl in enumerate(self.board_slices):
                idx_b = torch.as_tensor(np.asarray(sl, dtype=np.int64), device=self.device)
                board_masks[b, idx_b] = 1.0

        for epoch in range(self.max_epochs):
            self.net_.train()
            perm = torch.randperm(n, generator=gen).to(self.device)
            for s in range(steps_per_epoch):
                idx = perm[s * self.batch_size:(s + 1) * self.batch_size]
                if len(idx) < 2:  # BatchNorm needs >1 sample
                    continue
                xb = Xt[idx]
                if self.noise_std:
                    xb = xb + self.noise_std * torch.randn_like(xb)
                if self.feat_drop:
                    xb = xb * (torch.rand_like(xb) > self.feat_drop)
                if board_masks is not None:
                    if self.aug_board_gain_std:   # per-sample per-board gain jitter
                        g = 1.0 + self.aug_board_gain_std * torch.randn(
                            (len(idx), len(board_masks)), device=self.device)
                        xb = xb * ((g - 1.0) @ board_masks + 1.0)
                    if self.aug_board_drop_p:     # zero ONE random board's slice
                        k = torch.randint(len(board_masks), (len(idx),),
                                          device=self.device)
                        drop = (torch.rand(len(idx), device=self.device)
                                < self.aug_board_drop_p).to(xb.dtype)
                        xb = xb * (1.0 - drop[:, None] * board_masks[k])
                logit, reg = self.net_(xb)
                loss = (self.w_cls * F.cross_entropy(logit, yc[idx],
                                                     label_smoothing=self.label_smoothing)
                        + (1 - self.w_cls) * F.smooth_l1_loss(reg, yz[idx]))
                opt.zero_grad()
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.net_.parameters(), self.grad_clip)
                opt.step()
                sched.step()

            if have_val:
                pred = self._predict_xy(Xv)
                val_med = float(np.median(np.linalg.norm(yv_xy - pred, axis=1)))
                self.val_history_.append(val_med)
                if report_cb is not None:
                    report_cb(epoch, val_med)
                if val_med < best_val - 1e-6:
                    best_val, bad = val_med, 0
                    best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.best_val_ = best_val if have_val else None
        return self

    def _predict_xy(self, Xt):
        self.net_.eval()
        with torch.no_grad():
            _, reg = self.net_(Xt)
        return reg.cpu().numpy() * self.ystd_ + self.ymean_

    def predict(self, X):
        return self._predict_xy(self._t(np.asarray(X, dtype=np.float32)))

    def predict_proba(self, X):
        self.net_.eval()
        with torch.no_grad():
            logit, _ = self.net_(self._t(np.asarray(X, dtype=np.float32)))
            return F.softmax(logit, dim=1).cpu().numpy()

    def get_state(self):
        """Clone of the trained weights — feed to fit(..., init_state=) to fine-tune."""
        return {k: v.detach().clone() for k, v in self.net_.state_dict().items()}

    def to_cpu(self):
        """Move the network to CPU before pickling. MPS-resident tensors can
        segfault on unpickle in a fresh process; CPU tensors load anywhere and
        inference of this small MLP is plenty fast on CPU."""
        if self.net_ is not None:
            self.net_ = self.net_.to("cpu")
        self.device = torch.device("cpu")
        return self


def fit_seed_ensemble(make, X, y_xy, y_cls, X_val, y_val_xy, n_seeds=3):
    """Train n_seeds TorchLocalizers (make(seed)->estimator) and return them.
    Averaging their predict()/predict_proba() reduces variance (the honest win
    of ensembling at this data scale)."""
    return [make(s).fit(X, y_xy, y_cls, X_val, y_val_xy) for s in range(n_seeds)]


def ensemble_predict(models, X):
    return np.mean([m.predict(X) for m in models], axis=0)


def ensemble_proba(models, X):
    return np.mean([m.predict_proba(X) for m in models], axis=0)
