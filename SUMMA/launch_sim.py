import random
import numpy as np
import argparse
import struct

from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder

def parse_args():
    #[N,K]@[K,M]
    parser = argparse.ArgumentParser(description="SUMMA GEMM on simulator/hardware")
    parser.add_argument("--P", required=True, type=int, help="PEs rectangle size: P x P")
    parser.add_argument("--M", required=True, type=int, help="Input context length")
    parser.add_argument("--K", required=True, type=int, help="Word vector dimension")
    parser.add_argument("--N", required=True, type=int, help="Output dimension")
    parser.add_argument("--name", type=str, default="out",
                        help="Compiled artifact name (default: out)")
    parser.add_argument("--cmaddr", type=str, default=None,
                        help="CM address for hardware execution (auto-filled by SdkLauncher)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations (default: 2)")
    parser.add_argument("--repeat", type=int, default=10, help="Repeat iterations (default: 10)")
    parser.add_argument("--perf_only", action="store_true",
                        help="Skip data generation, H2D, and result D2H; only retrieve timing data")

    args = parser.parse_args()
    return args

def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)


def main():
    random.seed(2025)

    args = parse_args()

    P = args.P

    M = args.M
    K = args.K
    N = args.N

    Mt = M // P
    Kt = K // P
    Nt = N // P

    io_dtype = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR
    perf_only = args.perf_only

    if not perf_only:
        tensor_X = np.random.rand(M, K).astype(np.float16)
        tensor_W = np.random.rand(K, N).astype(np.float16)

    if args.cmaddr:
        runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    else:
        runner = SdkRuntime(args.name)
    runner.load()
    runner.run()

    symbol_time_memcpy = runner.get_id("time_memcpy")
    symbol_time_ref = runner.get_id("time_ref")

    if not perf_only:
        sym_X = runner.get_id("X")
        sym_W = runner.get_id("W")
        sym_res = runner.get_id("res")

        X1 = tensor_X.reshape(P, Mt, P, Kt)
        X2 = X1.transpose(0, 2, 3, 1)
        X3 = X2.reshape(P, P, Mt*Kt)
        X_u32 = input_array_to_u32(X3.ravel(), 1, 1)
        runner.memcpy_h2d(sym_X, X_u32, 0, 0, P, P, Mt*Kt, \
                          streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

        W1 = tensor_W.reshape(P, Kt, P, Nt)
        W2 = W1.transpose(0, 2, 1, 3)
        W3 = W2.reshape(P, P, Kt*Nt)
        W_u32 = input_array_to_u32(W3.ravel(), 1, 1)
        runner.memcpy_h2d(sym_W, W_u32, 0, 0, P, P, Kt*Nt, \
                          streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    # ---- Phase 1: correctness run (no warmup, 1 repeat) ----
    if not perf_only:
        runner.launch('init_task', nonblock=False)
        runner.launch('summa_host', np.int16(0), np.int16(1), nonblock=False)

        res3_1d_u32 = np.zeros(M*N, dtype=np.uint32)
        runner.memcpy_d2h(res3_1d_u32, sym_res, 0, 0, P, P, Mt*Nt, \
                          streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)
        res3_1d_fp16 = memcpy_view(res3_1d_u32, np.dtype(np.float16))
        res3 = res3_1d_fp16.reshape((P, P, Nt, Mt))
        res2 = res3.transpose(0, 3, 1, 2)
        res = res2.reshape(M, N)

    # ---- Phase 2: timing run ----
    runner.launch('init_task', nonblock=False)
    total_warmup_times, total_repeat_times = args.warmup, args.repeat
    runner.launch('summa_host', np.int16(total_warmup_times), np.int16(total_repeat_times), nonblock=False)

    # ---- D2H: timestamps ----
    time_memcpy_1d_f32 = np.zeros(P*P*3, dtype=np.float32)
    runner.memcpy_d2h(time_memcpy_1d_f32, symbol_time_memcpy, 0, 0, P, P, 3, streaming=False,
                    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    time_memcpy_hwl = np.reshape(time_memcpy_1d_f32, (P, P, 3), order='C')

    time_ref_1d_f32 = np.zeros(P*P*2, np.float32)
    runner.memcpy_d2h(time_ref_1d_f32, symbol_time_ref, 0, 0, P, P, 2, streaming=False,
                    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    time_ref_hwl = np.reshape(time_ref_1d_f32, (P, P, 2), order='C')

    runner.stop()

    # ---- Decode timestamps ----
    time_start = np.zeros((P, P)).astype(int)
    time_end = np.zeros((P, P)).astype(int)
    word = np.zeros(3).astype(np.uint16)
    for w in range(P):
        for h in range(P):
            hex_t0 = int(float_to_hex(time_memcpy_hwl[(h, w, 0)]), base=16)
            hex_t1 = int(float_to_hex(time_memcpy_hwl[(h, w, 1)]), base=16)
            hex_t2 = int(float_to_hex(time_memcpy_hwl[(h, w, 2)]), base=16)
            word[0] = hex_t0 & 0x0000ffff
            word[1] = (hex_t0 >> 16) & 0x0000ffff
            word[2] = hex_t1 & 0x0000ffff
            time_start[(h, w)] = make_u48(word)
            word[0] = (hex_t1 >> 16) & 0x0000ffff
            word[1] = hex_t2 & 0x0000ffff
            word[2] = (hex_t2 >> 16) & 0x0000ffff
            time_end[(h, w)] = make_u48(word)

    time_ref = np.zeros((P, P)).astype(int)
    word = np.zeros(3).astype(np.uint16)
    for w in range(P):
        for h in range(P):
            hex_t0 = int(float_to_hex(time_ref_hwl[(h, w, 0)]), base=16)
            hex_t1 = int(float_to_hex(time_ref_hwl[(h, w, 1)]), base=16)
            word[0] = hex_t0 & 0x0000ffff
            word[1] = (hex_t0 >> 16) & 0x0000ffff
            word[2] = hex_t1 & 0x0000ffff
            time_ref[(h, w)] = make_u48(word)

    for py in range(P):
        for px in range(P):
            time_ref[(py, px)] = time_ref[(py, px)] - (px + py)

    time_start = time_start - time_ref
    time_end = time_end - time_ref

    # ---- Verification (only for direct simulator use, skip for perf_only and hardware runs) ----
    if not perf_only and not args.cmaddr:
        expected_res = np.matmul(tensor_X, tensor_W)

        print("=" * 60)
        print("GEMM Verification (X @ W -> [M, N]):")
        print("-" * 60)
        print("Expected:")
        print(expected_res)
        print("Actual:")
        print(res)

        if np.allclose(res, expected_res, atol=1e-1):
            print("[PASS] GEMM result matches expected (atol=1e-1)")
        else:
            max_diff = np.max(np.abs(res.astype(np.float32) - expected_res.astype(np.float32)))
            print(f"[FAIL] GEMM result mismatch! Max abs diff = {max_diff}")
        print("=" * 60)

    # ---- Timing ----
    min_time_start = time_start.min()
    max_time_end = time_end.max()

    print(f"\nRepeat count: {total_repeat_times}")
    print(f"P: {P}, M: {M}, K: {K}, N: {N}")
    print(f"Mean cycle count: {np.mean(time_end - time_start)/total_repeat_times:.1f}")
    print(f"Max cycle count: {(max_time_end - min_time_start)/total_repeat_times:.1f}")

    freq_ghz = 1.1
    time_cost = (max_time_end - min_time_start) / total_repeat_times / (freq_ghz * 1e6)
    print(f"Time: {time_cost:.4f} ms")

if __name__ == "__main__":
    main()
