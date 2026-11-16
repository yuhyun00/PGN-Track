"""YAML configs with `_base_` inheritance and dotted `key=value` overrides."""
import copy
import os

import yaml


def merge(base: dict, update: dict) -> dict:
    """Recursively merge `update` into a copy of `base` (values of `update` win)."""
    out = copy.deepcopy(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str) -> dict:
    """Load a YAML file; `_base_` (a path or a list, relative to the file) is loaded first and merged under it."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    bases = cfg.pop('_base_', [])
    if isinstance(bases, str):
        bases = [bases]
    out = {}
    for b in bases:
        out = merge(out, load_config(os.path.join(os.path.dirname(os.path.abspath(path)), b)))
    return merge(out, cfg)


def parse_value(text: str):
    """Parse an override value as YAML; '1e-2'-style numbers (strings in YAML 1.1) become floats."""
    v = yaml.safe_load(text)
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            pass
    return v


def apply_overrides(cfg: dict, items) -> dict:
    """Apply overrides such as `motion.pgn.levels=[4]` in place and return the config."""
    for item in items or ():
        key, sep, text = item.partition('=')
        if not sep:
            raise ValueError(f'override must be key=value, got {item!r}')
        *parents, leaf = key.strip().split('.')
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f'cannot override {key}: {p} is not a section')
        node[leaf] = parse_value(text)
    return cfg


def check_keys(section: dict, allowed, name: str):
    """Raise on keys that a config section does not define."""
    bad = sorted(set(section) - set(allowed))
    if bad:
        raise KeyError(f'unknown key(s) in {name}: {", ".join(bad)}')
