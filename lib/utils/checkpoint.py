"""Robust checkpoint helpers: accept bare state_dicts AND full training checkpoints."""
import torch

_WRAPPER_KEYS = ("model", "state_dict", "model_state_dict", "net", "network",
                 "teacher_backbone", "backbone", "student_backbone")
_STRIP_PREFIXES = ("", "module.", "model.", "backbone.", "encoder.")


def torch_load_any(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # very old torch without weights_only
        return torch.load(path, map_location=map_location)


def extract_state_dict(ckpt):
    """Return a flat {param_name: tensor} dict from a bare state_dict or a full checkpoint."""
    if isinstance(ckpt, dict):
        for k in _WRAPPER_KEYS:
            if k in ckpt and isinstance(ckpt[k], dict):
                return extract_state_dict(ckpt[k])
    return ckpt


def load_pretrained_flexible(model, path, verbose=True):
    """Load every tensor whose name AND shape match (tries common prefix variants).
    Works with: bare state_dict, {'model': ...}, {'state_dict': ...}, DataParallel 'module.' keys,
    and the UniMiSS-3D encoder export. Returns (n_loaded, n_total)."""
    sd = extract_state_dict(torch_load_any(path))
    msd = model.state_dict()
    best_n, best = -1, None
    for strip in _STRIP_PREFIXES:
        cand = {(k[len(strip):] if strip and k.startswith(strip) else k): v for k, v in sd.items()
                if torch.is_tensor(v)}
        filt = {k: v for k, v in cand.items() if k in msd and msd[k].shape == v.shape}
        if len(filt) > best_n:
            best_n, best = len(filt), filt
    model.load_state_dict(best, strict=False)
    if verbose:
        print("[ckpt] loaded {}/{} tensors ({:.1f}%) from {}".format(
            best_n, len(msd), 100.0 * best_n / max(len(msd), 1), path))
        if best_n < 0.2 * len(msd):
            print("[ckpt] WARNING: <20% of the model was matched - wrong checkpoint / model config?")
    return best_n, len(msd)
