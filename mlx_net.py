"""Apple-MLX port of the PyTorch shared-trunk localizer (see ``torch_net.py``).

This is a faithful re-implementation of :class:`torch_net.TorchLocalizer` on
Apple's MLX framework, for a cross-framework reproducibility + throughput
experiment. The architecture, loss, optimizer, LR schedule, augmentation and
y-standardization are mirrored as closely as MLX allows.

Drop-in usage: same constructor params and the same
``fit(X, y_xy, y_cls, X_val, y_val_xy)`` / ``predict(X)`` / ``predict_proba(X)``
interface as ``TorchLocalizer`` (the per-board augmentation hooks are kept for
signature parity but are likewise OFF by default).

Unavoidable / intentional torch-vs-MLX differences
--------------------------------------------------
* **RNG is independent.** MLX has its own PRNG (``mlx.core.random``) seeded from
  ``seed``; CPU shuffling uses NumPy seeded from ``seed``. Numerical outputs will
  NOT match torch bit-for-bit even with the same seed — only the *behaviour*
  (it trains, error drops) is comparable, exactly as the experiment intends.
* **BatchNorm semantics.** ``norm="batch"`` maps to ``mlx.nn.BatchNorm`` (running
  stats, momentum=0.1, eps=1e-5) to match ``torch.nn.BatchNorm1d`` defaults as
  closely as MLX exposes; ``"layer"`` -> ``mlx.nn.LayerNorm``, ``"none"`` ->
  identity. Internals (parallel reductions, eps placement) differ slightly from
  torch, so BN running-mean/var will not be identical.
* **Dropout / GELU.** ``mlx.nn.Dropout`` and ``mlx.nn.GELU`` use the standard
  formulations; MLX GELU defaults to the exact (erf) form, matching torch's
  default ``nn.GELU()``. ``relu``/``silu`` map to ``mlx.nn.ReLU``/``mlx.nn.SiLU``.
* **AdamW.** ``mlx.optimizers.AdamW`` implements decoupled weight decay like
  torch's ``AdamW``; default betas/eps match (0.9, 0.999, 1e-8).
* **SmoothL1 / CE.** MLX exposes ``smooth_l1_loss`` (beta=1.0, same Huber knee as
  torch default) and ``cross_entropy`` (with ``label_smoothing``); used directly.
* **Lazy evaluation.** MLX is lazy — we call ``mx.eval`` after each optimizer
  step and force evaluation when reading metrics, so timing/throughput numbers
  reflect real compute.
* **Device.** MLX uses a unified-memory default device (GPU on Apple silicon);
  there is no explicit ``.to(device)`` per-tensor move like torch/MPS.
"""
import numpy as np

try:  # MLX is optional; the file must import-parse even when MLX is absent.
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    _HAS_MLX = True
except Exception:  # pragma: no cover - exercised only on non-MLX hosts
    mx = None
    nn = None
    optim = None
    _HAS_MLX = False


def _acts():
    return {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


def _norm_layer(norm, width):
    if norm == "batch":
        return nn.BatchNorm(width)
    if norm == "layer":
        return nn.LayerNorm(width)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm: {norm!r}")


class _Trunk(nn.Module):
    """Mirror of torch_net._Trunk: inp Linear -> depth blocks -> out head."""

    def __init__(self, d_in, width, depth, act, dropout, norm, residual):
        super().__init__()
        A = _acts()[act]
        self.inp = nn.Linear(d_in, width)
        self.blocks = [
            nn.Sequential(_norm_layer(norm, width), A(), nn.Dropout(dropout),
                          nn.Linear(width, width))
            for _ in range(depth)
        ]
        self.residual = residual
        self.out = nn.Sequential(_norm_layer(norm, width), A(), nn.Dropout(dropout))

    def __call__(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h) if self.residual else b(h)
        return self.out(h)


class _Net(nn.Module):
    """Shared trunk + two heads (cls logits, xy regression)."""

    def __init__(self, d_in, n_cls, width, depth, act, dropout, norm, residual):
        super().__init__()
        self.trunk = _Trunk(d_in, width, depth, act, dropout, norm, residual)
        self.cls = nn.Linear(width, n_cls)
        self.reg = nn.Linear(width, 2)

    def __call__(self, x):
        h = self.trunk(x)
        return self.cls(h), self.reg(h)


class MLXLocalizer:
    """Shared-trunk MLP estimator on Apple MLX — a drop-in alternative to
    :class:`torch_net.TorchLocalizer`.

    Expects ALREADY-SCALED float32 ``X`` (StandardScaler fit on the train fold
    only). ``y_xy`` is standardized internally (inverted at predict); ``y_cls``
    are integer class ids in ``[0, n_cls)``.

    See module docstring for the (unavoidable) torch-vs-MLX differences. The
    ``board_slices`` / ``aug_board_*`` hooks are kept for signature parity and
    default OFF (no behaviour when off, matching torch_net).
    """

    def __init__(self, n_cls, width=256, depth=2, act="gelu", dropout=0.2,
                 norm="batch", residual=False, lr=1e-3, weight_decay=1e-4,
                 batch_size=2048, max_epochs=300, patience=30, warmup_frac=0.05,
                 noise_std=0.0, feat_drop=0.0, label_smoothing=0.0, w_cls=0.5,
                 grad_clip=5.0, seed=0, verbose=False,
                 aug_board_gain_std=0.0, aug_board_drop_p=0.0, board_slices=None):
        if not _HAS_MLX:
            raise ImportError(
                "mlx is not installed; cannot construct MLXLocalizer. "
                "Install with `pip install mlx` on Apple silicon.")
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
        self.net_ = None
        self.val_history_ = []

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _a(x, dtype=None):
        """NumPy array -> mlx array (mlx defaults to float32 for float inputs)."""
        arr = np.asarray(x)
        if dtype is not None:
            arr = arr.astype(dtype)
        return mx.array(arr)

    def _loss_fn(self, net, xb, yc, yz):
        logit, reg = net(xb)
        cls = nn.losses.cross_entropy(
            logit, yc, label_smoothing=self.label_smoothing, reduction="mean")
        # smooth_l1_loss: beta/knee=1.0 like torch's default smooth_l1_loss
        regl = nn.losses.smooth_l1_loss(reg, yz, beta=1.0, reduction="mean")
        return self.w_cls * cls + (1.0 - self.w_cls) * regl

    # ---- training ----------------------------------------------------------
    def fit(self, X, y_xy, y_cls, X_val=None, y_val_xy=None, y_val_cls=None,
            report_cb=None, init_state=None, y_stats=None):
        mx.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y_xy = np.asarray(y_xy, dtype=np.float32)
        if y_stats is not None:
            self.ymean_, self.ystd_ = np.asarray(y_stats[0]), np.asarray(y_stats[1])
        else:
            self.ymean_ = y_xy.mean(0)
            self.ystd_ = y_xy.std(0) + 1e-8

        Xt = self._a(X)
        yz = self._a((y_xy - self.ymean_) / self.ystd_)
        yc = self._a(np.asarray(y_cls).astype(np.int32))
        have_val = X_val is not None and y_val_xy is not None
        if have_val:
            Xv = self._a(np.asarray(X_val, dtype=np.float32))
            yv_xy = np.asarray(y_val_xy, dtype=np.float32)

        self.net_ = _Net(X.shape[1], self.n_cls, self.width, self.depth, self.act,
                         self.dropout, self.norm, self.residual)
        if init_state is not None:  # warm-start (transfer/fine-tune)
            self.net_.update(init_state)
        mx.eval(self.net_.parameters())

        opt = optim.AdamW(learning_rate=self.lr, weight_decay=self.weight_decay)

        n = len(X)
        steps_per_epoch = max(1, n // self.batch_size)
        total_steps = steps_per_epoch * self.max_epochs
        warmup = max(1, int(total_steps * self.warmup_frac))

        def lr_at(step):  # linear warmup then cosine (identical to torch_net)
            if step < warmup:
                return self.lr * (step / warmup)
            prog = (step - warmup) / max(1, total_steps - warmup)
            return self.lr * 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))

        # per-board augmentation masks (only built when augmentation is on)
        board_masks = None
        if self.board_slices and (self.aug_board_gain_std or self.aug_board_drop_p):
            bm = np.zeros((len(self.board_slices), X.shape[1]), dtype=np.float32)
            for b, sl in enumerate(self.board_slices):
                bm[b, np.asarray(sl, dtype=np.int64)] = 1.0
            board_masks = self._a(bm)

        loss_and_grad = nn.value_and_grad(self.net_, self._loss_fn)

        best_val, best_state, bad = np.inf, None, 0
        global_step = 0
        for epoch in range(self.max_epochs):
            self.net_.train()
            perm = rng.permutation(n)
            for s in range(steps_per_epoch):
                idx = perm[s * self.batch_size:(s + 1) * self.batch_size]
                if len(idx) < 2:  # BatchNorm needs >1 sample
                    continue
                idx_m = mx.array(idx.astype(np.int32))
                xb = Xt[idx_m]
                if self.noise_std:
                    xb = xb + self.noise_std * mx.random.normal(xb.shape)
                if self.feat_drop:
                    keep = (mx.random.uniform(shape=xb.shape) > self.feat_drop)
                    xb = xb * keep.astype(xb.dtype)
                if board_masks is not None:
                    if self.aug_board_gain_std:   # per-sample per-board gain jitter
                        g = 1.0 + self.aug_board_gain_std * mx.random.normal(
                            (len(idx), board_masks.shape[0]))
                        xb = xb * ((g - 1.0) @ board_masks + 1.0)
                    if self.aug_board_drop_p:     # zero ONE random board's slice
                        k = mx.random.randint(0, board_masks.shape[0], (len(idx),))
                        drop = (mx.random.uniform(shape=(len(idx),))
                                < self.aug_board_drop_p).astype(xb.dtype)
                        xb = xb * (1.0 - drop[:, None] * board_masks[k])

                opt.learning_rate = lr_at(global_step)
                loss, grads = loss_and_grad(self.net_, xb, yc[idx_m], yz[idx_m])
                if self.grad_clip:
                    grads, _ = optim.clip_grad_norm(grads, self.grad_clip)
                opt.update(self.net_, grads)
                mx.eval(self.net_.parameters(), opt.state)
                global_step += 1

            if have_val:
                pred = self._predict_xy(Xv)
                val_med = float(np.median(np.linalg.norm(yv_xy - pred, axis=1)))
                self.val_history_.append(val_med)
                if report_cb is not None:
                    report_cb(epoch, val_med)
                if val_med < best_val - 1e-6:
                    best_val, bad = val_med, 0
                    best_state = self._clone_params()
                else:
                    bad += 1
                    if bad >= self.patience:
                        break

        if best_state is not None:
            self.net_.update(best_state)
            mx.eval(self.net_.parameters())
        self.best_val_ = best_val if have_val else None
        return self

    # ---- inference ---------------------------------------------------------
    def _predict_xy(self, Xt):
        self.net_.eval()
        _, reg = self.net_(Xt)
        mx.eval(reg)
        return np.array(reg) * self.ystd_ + self.ymean_

    def predict(self, X):
        return self._predict_xy(self._a(np.asarray(X, dtype=np.float32)))

    def predict_proba(self, X):
        self.net_.eval()
        logit, _ = self.net_(self._a(np.asarray(X, dtype=np.float32)))
        proba = mx.softmax(logit, axis=1)
        mx.eval(proba)
        return np.array(proba)

    # ---- state (transfer/fine-tune + pickling) -----------------------------
    def _clone_params(self):
        """Deep-ish clone of trained params as a flat dict of mlx arrays."""
        from mlx.utils import tree_map
        return tree_map(lambda p: mx.array(np.array(p)), self.net_.parameters())

    def get_state(self):
        """Clone of the trained weights — feed to fit(..., init_state=)."""
        return self._clone_params()

    def to_cpu(self):
        """Parity no-op: MLX uses unified memory, so there is no GPU->CPU move
        needed before pickling (kept for interface symmetry with TorchLocalizer)."""
        return self


def fit_seed_ensemble(make, X, y_xy, y_cls, X_val, y_val_xy, n_seeds=3):
    """Train n_seeds MLXLocalizers (make(seed)->estimator) and return them."""
    return [make(s).fit(X, y_xy, y_cls, X_val, y_val_xy) for s in range(n_seeds)]


def ensemble_predict(models, X):
    return np.mean([m.predict(X) for m in models], axis=0)


def ensemble_proba(models, X):
    return np.mean([m.predict_proba(X) for m in models], axis=0)
