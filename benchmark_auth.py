import asyncio
import hmac
import time

def setup_baseline(num_keys=1000, key_length=32):
    keys = [f"valid_key_{i:020d}" for i in range(num_keys)]
    return keys

def setup_optimized(num_keys=1000, key_length=32):
    keys = [f"valid_key_{i:020d}".encode() for i in range(num_keys)]
    return keys

def run_baseline(keys, target_token):
    token_bytes = target_token.encode()
    # Worst case, no match
    return any(hmac.compare_digest(token_bytes, k.encode()) for k in keys)

def run_optimized(keys_bytes, target_token):
    token_bytes = target_token.encode()
    return any(hmac.compare_digest(token_bytes, k) for k in keys_bytes)

def benchmark():
    num_keys = 100
    iterations = 10000

    baseline_keys = setup_baseline(num_keys)
    optimized_keys = setup_optimized(num_keys)
    target_token = "invalid_token_that_wont_match"

    start = time.perf_counter()
    for _ in range(iterations):
        run_baseline(baseline_keys, target_token)
    baseline_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        run_optimized(optimized_keys, target_token)
    optimized_time = time.perf_counter() - start

    print(f"Baseline (100 keys, {iterations} requests): {baseline_time:.4f}s")
    print(f"Optimized (100 keys, {iterations} requests): {optimized_time:.4f}s")
    print(f"Improvement: {(baseline_time - optimized_time) / baseline_time * 100:.2f}%")

if __name__ == "__main__":
    benchmark()
