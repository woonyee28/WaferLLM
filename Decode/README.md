# Decode WSE-3-GQA

## Overview

Grouped-Query Attention (GQA) decode kernel for transformer inference on Cerebras WSE-3. Extends the base WSE-3 decode with compact KV caches, interleaved Q layout (Option C), and KV-head-scoped all-reduce.

## Platform

- **Cerebras SDK version**: 1.4
- **Cerebras ML Software version**: 2.5
- **Hardware**: WSE-3 only

## Configuration

Configuration files are in `model_config/`. Example (`gqa_test.json`):

```json
{
    "P": 8,
    "group_num": 4,
    "bsz": 2,
    "dim": 16,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 4,
    "max_seq_len": 16,
    "prefill_len": 8,
    "ffn_dim": 64
}
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `P` | PE grid dimension (creates P x P grid) |
| `group_num` | Number of PE groups for communication |
| `bsz` | Batch size |
| `dim` | Model hidden dimension |
| `n_heads` | Number of query attention heads |
| `n_kv_heads` | Number of key-value heads (GQA groups) |
| `head_dim` | Dimension per attention head |
| `max_seq_len` | Maximum sequence length (KV cache capacity) |
| `prefill_len` | Number of prefilled tokens in KV cache |
| `ffn_dim` | Feed-forward network hidden dimension |

**Constraints:**
- All dimensions must be divisible by `P`
- `group_num` must be a multiple of both `n_heads` and `n_kv_heads`
- `prefill_len` must be divisible by `P`
- `max_seq_len >= prefill_len`

## Run with Simulator

```bash
bash run_sim.sh model_config/gqa_test.json
```

## Run on WSE-3 Hardware

### Direct execution (via SdkCompiler + SdkRuntime)

```bash
# Compile + run on appliance simulator
bash run_wse3.sh model_config/gqa_test.json true

# Compile + run on real WSE-3
bash run_wse3.sh model_config/gqa_test.json false
```

### Via SdkLauncher (compile locally, dispatch to appliance)

```bash
# Step 1: Compile (uses cslc)
python compile.py model_config/gqa_test.json false

# Step 2: Dispatch to appliance
python run_sdk_launcher.py --config model_config/gqa_test.json

# Or do both in one command:
bash run_launcher.sh model_config/gqa_test.json

# With appliance simulator:
bash run_launcher.sh model_config/gqa_test.json true
```

### Compilation only

```bash
# For simulator (small fabric)
python compile.py model_config/gqa_test.json true

# For hardware (full 762x1172 fabric, 4 channels)
python compile.py model_config/gqa_test.json false
```

## File Structure

```
WSE-3-GQA/
├── src/
│   ├── layout.csl              # PE grid configuration
│   ├── decode.csl              # GQA decode kernel
│   └── comm_lib/               # Communication library
│       ├── comm_layout.csl     # Routing topology (incl. KV-head-scoped routes)
│       └── comm_pe.csl         # PE-level comm primitives
├── model_config/               # JSON config files
├── compile.py                  # Compilation script (cslc subprocess)
├── launch_sim.py               # Simulator host code
├── launch_wse3.py              # Hardware host code (SdkRuntime)
├── run_sdk_launcher.py         # SdkLauncher dispatch script
├── run_sim.sh                  # Simulator workflow
├── run_wse3.sh                 # Hardware workflow (direct)
├── run_launcher.sh             # Hardware workflow (SdkLauncher)
└── validate.py                 # Output validation
```
