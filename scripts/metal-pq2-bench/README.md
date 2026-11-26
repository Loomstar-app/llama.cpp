Standalone Metal probes for the PQ2_0 decode mat-vec on Apple GPUs (measured on M2 Pro 16-core).

    clang -O2 -fobjc-arc -framework Metal -framework Foundation bandwidth.m -o bw && ./bw
    clang -O2 -fobjc-arc -framework Metal -framework Foundation harness.m -o h && ./h 17408 5120 kernels.metal

harness.m streams ~1.6 GB of copies so the SLC never holds the weights, and checks every kernel
against a double-precision reference. Variants: stock (fork kernel), stock_load (loads only),
stock_math (math only, no weight loads), rp* (repacked qs/d layout, uint2/uint4 loads), alt_* (cheaper decodes).
