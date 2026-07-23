"""Export a starmelee torch checkpoint for the standalone exe.

Usage (from the repo root):
    python ocean/starmelee/export_policy.py [checkpoint|latest] [out.bin]

Writes resources/starmelee/starmelee_policy.bin by default. The format is
consumed by the inference code in ocean/starmelee/starmelee.c: keep both in
sync. Layout ('SMP1', little-endian):
    int32 magic, obs_size, hidden, num_layers, num_heads, nvec[num_heads]
    f32 enc_w[hidden][obs_size], enc_b[hidden]
    f32 gru_w[3*hidden][hidden]      (per layer, in order)
    f32 dec_w[sum(nvec)][hidden], dec_b[sum(nvec)]
"""
import glob
import os
import struct
import sys

import numpy as np
import torch

MAGIC = 0x31504D53  # 'SMP1'
NVEC = (2, 2, 2)


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
    if dec_w.shape != (sum(NVEC), hidden):
        sys.exit(f'decoder shape {dec_w.shape} does not match nvec {NVEC}')

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'wb') as f:
        f.write(struct.pack('<5i', MAGIC, obs_size, hidden, len(layer_keys), len(NVEC)))
        f.write(struct.pack(f'<{len(NVEC)}i', *NVEC))
        f.write(enc_w.tobytes())
        f.write(enc_b.tobytes())
        for k in layer_keys:
            gw = sd[k].numpy().astype(np.float32)
            if gw.shape != (3 * hidden, hidden):
                sys.exit(f'{k} shape {gw.shape} unexpected')
            f.write(gw.tobytes())
        f.write(dec_w.tobytes())
        f.write(dec_b.tobytes())

    print(f'Exported {ckpt}')
    print(f'  -> {out} (obs {obs_size}, hidden {hidden}, layers {len(layer_keys)})')


if __name__ == '__main__':
    main()
