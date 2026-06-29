"""Small CPU smoke test for the MLX port (mlx_net.MLXLocalizer).

Skips gracefully when MLX is not installed. This is a *behavioural* sanity check
only — it does NOT compare numerically to the torch model (independent RNG /
framework). It verifies the model trains and the public interface works:

* fit() then predict() returns finite (n, 2) coordinates,
* predict_proba() is a proper distribution (sums to 1 over classes),
* training reduces regression error vs an untrained baseline.
"""
import numpy as np
import pytest

pytest.importorskip("mlx.core")  # skip cleanly when MLX is absent

from mlx_net import MLXLocalizer


def _toy(n=500, d=20, n_cls=4, seed=0):
    """Synthetic, learnable task: cell id -> a clustered (x,y) target, with the
    features being a noisy linear-ish embedding of the class so an MLP can fit it.

    The class->feature projection ``W`` is FIXED (seed-independent) so that a
    train split (seed=0) and a val split (seed=1) share the same underlying
    feature->class->position mapping — mirroring a real dataset where train and
    test obey the same physics. ``seed`` only varies the class draws and noise,
    not the task itself; using it for ``W`` too would make the two splits
    unrelated and unlearnable (a model fit on one could never generalize)."""
    rng = np.random.default_rng(seed)
    y_cls = rng.integers(0, n_cls, size=n).astype(np.int64)
    # per-class centers in a 2x2 grid of "room" coords (meters-ish)
    centers_xy = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.0, 3.0]],
                          dtype=np.float32)[:n_cls]
    y_xy = centers_xy[y_cls] + rng.normal(scale=0.15, size=(n, 2)).astype(np.float32)
    # features: random projection of a one-hot class + noise (already "scaled").
    # W is drawn from a fixed seed so all splits share the same mapping.
    W = np.random.default_rng(12345).normal(size=(n_cls, d)).astype(np.float32)
    X = W[y_cls] + rng.normal(scale=0.5, size=(n, d)).astype(np.float32)
    X = X.astype(np.float32)
    return X, y_xy, y_cls


def test_mlx_localizer_trains_and_predicts():
    n, d, n_cls = 500, 20, 4
    X, y_xy, y_cls = _toy(n, d, n_cls, seed=0)
    Xv, yv_xy, yv_cls = _toy(160, d, n_cls, seed=1)

    # tiny/fast config — enough epochs to clearly beat the untrained baseline
    model = MLXLocalizer(
        n_cls=n_cls, width=64, depth=2, dropout=0.1, norm="batch",
        lr=3e-3, weight_decay=1e-4, batch_size=64, max_epochs=60,
        patience=60, warmup_frac=0.1, noise_std=0.01, feat_drop=0.05,
        w_cls=0.5, seed=0,
    )

    # untrained baseline error: predict the global-mean (x,y) for every sample
    base_pred = np.tile(y_xy.mean(0), (len(Xv), 1))
    base_err = float(np.median(np.linalg.norm(yv_xy - base_pred, axis=1)))

    model.fit(X, y_xy, y_cls, X_val=Xv, y_val_xy=yv_xy)

    pred = model.predict(Xv)
    assert pred.shape == (len(Xv), 2)
    assert np.all(np.isfinite(pred))

    proba = model.predict_proba(Xv)
    assert proba.shape == (len(Xv), n_cls)
    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert np.all(proba >= -1e-6)

    trained_err = float(np.median(np.linalg.norm(yv_xy - pred, axis=1)))
    # training must meaningfully beat the constant-mean baseline
    assert trained_err < base_err * 0.7, (
        f"training did not reduce error enough: trained={trained_err:.3f} "
        f"baseline={base_err:.3f}")


def test_predict_proba_argmax_beats_chance():
    """Cls head should learn *something* — accuracy clearly above 1/n_cls."""
    n, d, n_cls = 500, 20, 4
    X, y_xy, y_cls = _toy(n, d, n_cls, seed=2)
    Xv, yv_xy, yv_cls = _toy(200, d, n_cls, seed=3)

    model = MLXLocalizer(
        n_cls=n_cls, width=64, depth=2, dropout=0.1, norm="batch",
        lr=3e-3, batch_size=64, max_epochs=60, patience=60,
        warmup_frac=0.1, w_cls=0.7, seed=0,
    )
    model.fit(X, y_xy, y_cls, X_val=Xv, y_val_xy=yv_xy)

    acc = float((model.predict_proba(Xv).argmax(1) == yv_cls).mean())
    assert acc > 1.0 / n_cls + 0.15, f"cls accuracy {acc:.3f} not above chance"
