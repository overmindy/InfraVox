"""State/mask unit test against pinned Kyutai RingKVCache, without LM weights.

Extract only the two official cache classes so the test needs torch but not
unrelated codec/model dependencies. This is not a numerical LM deployment test.
"""
import argparse
import ast
import typing
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=1000)
    args = parser.parse_args()
    if args.capacity < 2:
        raise ValueError("Capacity must be >= 2")
    source = ast.parse(args.transformer.read_text())
    classes = [node for node in source.body if isinstance(node, ast.ClassDef)
               and node.name in ("KVCacheResult", "RingKVCache")]
    if len(classes) != 2:
        raise ValueError("Pinned upstream cache definitions changed")
    namespace = {"torch": torch, "tp": typing}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(args.transformer), "exec"), namespace)
    upstream = namespace["RingKVCache"](1, 2, 4, args.capacity, device=torch.device("cpu"), dtype=torch.float32)
    expected_k = torch.zeros(1, 2, args.capacity, 4)
    expected_v = torch.zeros_like(expected_k)
    pointer = upstream.cache.data_ptr()
    boundaries = {0, 1, args.capacity - 2, args.capacity - 1, args.capacity, args.capacity + 1}
    for position in range(args.capacity + 2):
        key = torch.arange(8, dtype=torch.float32).reshape(1, 2, 1, 4) + position * 10
        value = key + 0.5
        result = upstream.complete(key, value, torch.ones(1, dtype=torch.bool))
        slot = position % args.capacity
        expected_k[:, :, slot:slot + 1] = key
        expected_v[:, :, slot:slot + 1] = value
        torch.testing.assert_close(result.keys, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(result.values, expected_v, rtol=0, atol=0)
        assert upstream.cache.data_ptr() == pointer
        slots = torch.arange(args.capacity)
        delta = slots - slot
        absolute = torch.where(delta <= 0, position + delta, position + delta - args.capacity)
        valid = (absolute >= 0) & ((position - absolute) < args.capacity)
        torch.testing.assert_close(valid, result.positions[0] >= 0)
        torch.testing.assert_close(absolute[valid], result.positions[0][valid])
        if position in boundaries:
            print(f"PASS position={position} slot={slot} valid={int(valid.sum())} oldest={int(absolute[valid].min())}")
    upstream.reset(torch.ones(1, dtype=torch.bool))
    result = upstream.complete(torch.ones(1, 2, 1, 4), torch.ones(1, 2, 1, 4), torch.ones(1, dtype=torch.bool))
    assert int((result.positions >= 0).sum()) == 1
    assert upstream.cache.data_ptr() == pointer
    print("PASS: upstream scatter storage, absolute-position mask, wrap and reset")


if __name__ == "__main__":
    main()
