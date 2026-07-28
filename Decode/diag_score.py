"""diag_score.py — diagnose the decode SCORE stage from the SAVED --dump-intermediates arrays.

After the RoPE fix, post-rope Q/K match HF but score_post_reduce (logits) is only cosine ~0.58.
This inspects the saved score grids (dump_intermediates/) to localize WHERE, with no device/recompile:

  [1] softmax row-sums per head  -> ~1.0 means the reconstruct mapping is right & softmax is a valid
      distribution; far from 1.0 means recon_scores maps to the wrong elements (my bug, not the kernel).
  [2] cross-px consistency within each kv-head -> score_post_reduce is AFTER the kv-head X-reduce, so
      all pes_p_kv_head px in a kv-head must be IDENTICAL. Large diff => the reduce didn't broadcast
      (or each px still holds only its own segment partial) = the kernel reduce is the bug.
  [3]/[4] per-head cosine + head-0 dev-vs-ref logits & softmax argmax -> the error pattern.

    python diag_score.py --config model_config/llama8B_block0_p64_attn.json
"""
import os
import json
import argparse
import numpy as np


def cos_(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="model_config/llama8B_block0_p64_attn.json")
    ap.add_argument("--dump-dir", default="dump_intermediates")
    a = ap.parse_args()
    c = json.load(open(a.config))
    P, n_heads, n_kv, head_dim, pos = c["P"], c["n_heads"], c["n_kv_heads"], c["head_dim"], c["prefill_len"]
    gqa = n_heads // n_kv
    pes_p_kv_head = P // n_kv
    iter_num = pos // P + 1

    D = a.dump_dir
    spr = np.load(os.path.join(D, "raw_score_post_reduce.npy")).astype(np.float64)   # [P,P,8] logits
    sm = np.load(os.path.join(D, "raw_score.npy")).astype(np.float64)                # [P,P,8] softmax
    ref_logits = np.load(os.path.join(D, "ref_logits.npy"))
    ref_attn = np.load(os.path.join(D, "ref_attn.npy"))

    def recon(grid):                                   # -> [n_heads, pos+1]; token t on py=t%P, slot=t//P
        out = np.zeros((n_heads, pos + 1))
        for h in range(n_heads):
            kvh, g = h // gqa, h % gqa
            px = kvh * pes_p_kv_head
            for t in range(pos + 1):
                out[h, t] = grid[t % P, px, g * iter_num + (t // P)]
        return out

    dev_logits, dev_sm = recon(spr), recon(sm)
    print(f"P={P} pos={pos} iter_num={iter_num} gqa={gqa} pes_p_kv_head={pes_p_kv_head}\n")

    rs = dev_sm.sum(axis=1)
    print(f"[1] device softmax row-sums: min={rs.min():.4f} max={rs.max():.4f}  (want ~1.0)")

    worst = 0.0
    for kvh in range(n_kv):
        base = kvh * pes_p_kv_head
        for g in range(gqa):
            col0 = spr[:, base, g * iter_num:g * iter_num + iter_num]
            for off in range(1, pes_p_kv_head):
                worst = max(worst, np.abs(spr[:, base + off, g * iter_num:g * iter_num + iter_num] - col0).max())
    print(f"[2] cross-px max diff within kv-heads: {worst:.4e}  (want ~0 if the X-reduce broadcasts)")

    ch = np.array([cos_(dev_logits[h], ref_logits[h]) for h in range(n_heads)])
    print(f"[3] per-head logit cosine: min={ch.min():.3f} mean={ch.mean():.3f} max={ch.max():.3f}"
          f"  | cos>0.99: {(ch > 0.99).sum()}/{n_heads}  cos<0.5: {(ch < 0.5).sum()}/{n_heads}")

    print("\n[4] head 0 logits (dev vs ref) at sample tokens:")
    for t in [0, 1, 2, 3, 31, 32, 62, 63, 64]:
        print(f"    t={t:2d}  dev={dev_logits[0, t]:+9.3f}  ref={ref_logits[0, t]:+9.3f}")
    print(f"    head-0 softmax argmax: dev={dev_sm[0].argmax()}  ref={ref_attn[0].argmax()}")
    print(f"    head-1 softmax argmax: dev={dev_sm[1].argmax()}  ref={ref_attn[1].argmax()}")

    # [5] Pre-reduce partials: manually sum the pes_p_kv_head segments ourselves. If that matches the
    # reference, the GEMV is correct and the on-device kv-head REDUCE is the bug. Also test whether the
    # device's g>0 result is just the base-PE (segment-0) partial (i.e. the reduce didn't sum groups 1+).
    gemv_path = os.path.join(D, "raw_score_post_gemv.npy")
    if os.path.exists(gemv_path):
        spg = np.load(gemv_path).astype(np.float64)          # [P,P,8] partials, pre-reduce
        alpha = 1.0 / np.sqrt(head_dim)
        man = np.zeros((n_heads, pos + 1))                   # our own full sum over segments
        seg0 = np.zeros((n_heads, pos + 1))                  # base-PE (segment 0) partial only
        for h in range(n_heads):
            kvh, g = h // gqa, h % gqa
            for t in range(pos + 1):
                py, slot = t % P, t // P
                man[h, t] = alpha * sum(spg[py, kvh * pes_p_kv_head + s, g * iter_num + slot]
                                        for s in range(pes_p_kv_head))
                seg0[h, t] = alpha * spg[py, kvh * pes_p_kv_head + 0, g * iter_num + slot]
        cm = np.array([cos_(man[h], ref_logits[h]) for h in range(n_heads)])
        print(f"\n[5] MANUAL segment-sum of score_post_gemv vs ref logits: "
              f"min={cm.min():.3f} mean={cm.mean():.3f} max={cm.max():.3f}  | >0.99: {(cm > 0.99).sum()}/{n_heads}")
        print("    (all ~1.0 => GEMV partials are correct => the kv-head REDUCE is the bug)")
        print("    per-GQA-group: cos(device post_reduce, our full sum) & cos(device, base-seg0-only):")
        for g in range(gqa):
            hs = [h for h in range(n_heads) if h % gqa == g]
            dev = np.concatenate([dev_logits[h] for h in hs])
            print(f"      g={g}:  cos(dev, full)={cos_(dev, np.concatenate([man[h] for h in hs])):+.3f}"
                  f"   cos(dev, seg0)={cos_(dev, np.concatenate([seg0[h] for h in hs])):+.3f}")
    else:
        print("\n[5] (raw_score_post_gemv.npy not found — re-run run_verify.sh --dump-intermediates "
              "after pulling to capture the pre-reduce partials)")


if __name__ == "__main__":
    main()
