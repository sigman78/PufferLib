"""Export a starmelee checkpoint for the standalone exe.

Usage (from the repo root):
    python ocean/starmelee/export_policy.py [checkpoint|latest] [out.bin]

Accepts BOTH backends' checkpoints and auto-detects which it got:
  - torch (--slowly) checkpoints are zip archives (torch.save state dict);
  - native CUDA trainer checkpoints are a raw f32 dump of the flat
    master_weights arena (save_weights in src/bindings.cu), laid out in
    registration order (models.cu policy weight reg): encoder weight
    [hidden][obs], decoder weight [sum(nvec)+1][hidden] whose LAST row is
    the fused value head, then per-layer MinGRU weights [3*hidden][hidden].
    The native arch has no biases; zeros are exported (the forward math in
    starmelee.c is then identical: same matmuls, same MinGRU gate).

Writes resources/starmelee/starmelee_policy.bin by default. The format is
consumed by the inference code in ocean/starmelee/starmelee.c: keep both in
sync. Layout ('SMP1', little-endian):
    int32 magic, obs_size, hidden, num_layers, num_heads, nvec[num_heads]
    f32 enc_w[hidden][obs_size], enc_b[hidden]
    f32 gru_w[3*hidden][hidden]      (per layer, in order)
    f32 dec_w[sum(nvec)][hidden], dec_b[sum(nvec)]
"""
import configparser
import glob
import os
import struct
import sys

import numpy as np

MAGIC = 0x31504D53  # 'SMP1'
NVEC = (2, 2, 2, 2)  # left, right, engine, fire


def load_torch(ckpt):
    import torch
    sd = torch.load(ckpt, map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in sd.items()}

    enc_w = sd['encoder.encoder.weight'].numpy().astype(np.float32)
    enc_b = sd['encoder.encoder.bias'].numpy().astype(np.float32)
    layer_keys = sorted(
        (k for k in sd if k.startswith('network.layers.') and k.endswith('.weight')),
        key=lambda k: int(k.split('.')[2]))
    dec_w = sd['decoder.decoder.weight'].numpy().astype(np.float32)
    dec_b = sd['decoder.decoder.bias'].numpy().astype(np.float32)

    hidden, obs_size = enc_w.shape
    gru_ws = []
    for k in layer_keys:
        gw = sd[k].numpy().astype(np.float32)
        if gw.shape != (3 * hidden, hidden):
            sys.exit(f'{k} shape {gw.shape} unexpected')
        gru_ws.append(gw)
    return enc_w, enc_b, gru_ws, dec_w, dec_b


def load_native(ckpt):
    # hidden/num_layers come from the ini's [policy] section (the native
    # trainer got them from the same place); obs_size is solved from the
    # element count, which doubles as a layout check.
    ini = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
    ini.read(os.path.join('config', 'starmelee.ini'))
    hidden = ini.getint('policy', 'hidden_size')
    layers = ini.getint('policy', 'num_layers')

    flat = np.fromfile(ckpt, dtype=np.float32)
    od1 = sum(NVEC) + 1  # +1: fused value head row
    rest = flat.size - od1 * hidden - layers * 3 * hidden * hidden
    if rest <= 0 or rest % hidden != 0:
        sys.exit(f'{ckpt}: {flat.size} floats does not fit the native layout '
                 f'(hidden {hidden}, layers {layers}, nvec {NVEC}) - '
                 f'stale checkpoint from an older obs/action space?')
    obs_size = rest // hidden

    off = 0
    enc_w = flat[off:off + hidden * obs_size].reshape(hidden, obs_size); off += hidden * obs_size
    dec_full = flat[off:off + od1 * hidden].reshape(od1, hidden); off += od1 * hidden
    gru_ws = []
    for _ in range(layers):
        gru_ws.append(flat[off:off + 3 * hidden * hidden].reshape(3 * hidden, hidden))
        off += 3 * hidden * hidden

    dec_w = dec_full[:sum(NVEC)]  # drop the value row: the exe only acts
    enc_b = np.zeros(hidden, dtype=np.float32)
    dec_b = np.zeros(sum(NVEC), dtype=np.float32)
    return enc_w, enc_b, gru_ws, dec_w, dec_b


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else 'latest'
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        'resources', 'starmelee', 'starmelee_policy.bin')

    if ckpt == 'latest':
        candidates = glob.glob(
            os.path.join('checkpoints', 'starmelee', '**', '*.bin'), recursive=True)
        if not candidates:
            sys.exit('No checkpoints under checkpoints/starmelee/')
        ckpt = max(candidates, key=os.path.getctime)

    with open(ckpt, 'rb') as f:
        is_torch = f.read(2) == b'PK'  # torch.save produces a zip archive
    kind = 'torch' if is_torch else 'native'
    enc_w, enc_b, gru_ws, dec_w, dec_b = (load_torch if is_torch else load_native)(ckpt)

    hidden, obs_size = enc_w.shape
    if dec_w.shape != (sum(NVEC), hidden):
        sys.exit(f'decoder shape {dec_w.shape} does not match nvec {NVEC}')

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'wb') as f:
        f.write(struct.pack('<5i', MAGIC, obs_size, hidden, len(gru_ws), len(NVEC)))
        f.write(struct.pack(f'<{len(NVEC)}i', *NVEC))
        f.write(np.ascontiguousarray(enc_w).tobytes())
        f.write(enc_b.tobytes())
        for gw in gru_ws:
            f.write(np.ascontiguousarray(gw).tobytes())
        f.write(np.ascontiguousarray(dec_w).tobytes())
        f.write(dec_b.tobytes())

    print(f'Exported {ckpt} ({kind})')
    print(f'  -> {out} (obs {obs_size}, hidden {hidden}, layers {len(gru_ws)})')


if __name__ == '__main__':
    main()
