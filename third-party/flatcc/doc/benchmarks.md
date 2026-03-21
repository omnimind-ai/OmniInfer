# Benchmarks

Benchmarks are defined for raw C structs, Googles `flatc` generated C++
and the `flatcc` compilers C ouput.

These can be run with:

    scripts/benchmark.sh

and requires a C++ compiler installed - the benchmark for flatcc alone can be
run with:

    test/benchmark/benchflatcc/run.sh

this only requires a system C compiler (cc) to be installed (and
flatcc's build environment).

A summary for OS-X 2.2 GHz Haswell core-i7 is found below. Generated
files for OS-X and Ubuntu are found in the benchmark folder.

The benchmarks use the same schema and dataset as Google FPL's
FlatBuffers benchmark.

In summary, 1 million iterations runs at about 500-540MB/s at 620-700
ns/op encoding buffers and 29-34ns/op traversing buffers. `flatc` and
`flatcc` are close enough in performance for it not to matter much.
`flatcc` is a bit faster encoding but it is likely due to less memory
allocation. Throughput and time per operatin are of course very specific
to this test case.

Generated JSON parser/printer shown below, for flatcc only but for OS-X
and Linux.


## operation: flatbench for raw C structs encode (optimized)
    elapsed time: 0.055 (s)
    iterations: 1000000
    size: 312 (bytes)
    bandwidth: 5665.517 (MB/s)
    throughput in ops per sec: 18158707.100
    throughput in 1M ops per sec: 18.159
    time per op: 55.070 (ns)

## operation: flatbench for raw C structs decode/traverse (optimized)
    elapsed time: 0.012 (s)
    iterations: 1000000
    size: 312 (bytes)
    bandwidth: 25978.351 (MB/s)
    throughput in ops per sec: 83263946.711
    throughput in 1M ops per sec: 83.264
    time per op: 12.010 (ns)

## operation: flatc for C++ encode (optimized)
    elapsed time: 0.702 (s)
    iterations: 1000000
    size: 344 (bytes)
    bandwidth: 490.304 (MB/s)
    throughput in ops per sec: 1425301.380
    throughput in 1M ops per sec: 1.425
    time per op: 701.606 (ns)

## operation: flatc for C++ decode/traverse (optimized)
    elapsed time: 0.029 (s)
    iterations: 1000000
    size: 344 (bytes)
    bandwidth: 11917.134 (MB/s)
    throughput in ops per sec: 34642832.398
    throughput in 1M ops per sec: 34.643
    time per op: 28.866 (ns)


## operation: flatcc for C encode (optimized)
    elapsed time: 0.626 (s)
    iterations: 1000000
    size: 336 (bytes)
    bandwidth: 536.678 (MB/s)
    throughput in ops per sec: 1597255.277
    throughput in 1M ops per sec: 1.597
    time per op: 626.074 (ns)

## operation: flatcc for C decode/traverse (optimized)
    elapsed time: 0.029 (s)
    iterations: 1000000
    size: 336 (bytes)
    bandwidth: 11726.930 (MB/s)
    throughput in ops per sec: 34901577.551
    throughput in 1M ops per sec: 34.902
    time per op: 28.652 (ns)

## JSON benchmark

*Note: this benchmark is only available for `flatcc`. It uses the exact
same data set as above.*

The benchmark uses Grisu3 floating point parsing and printing algorithm
with exact fallback to strtod/sprintf when the algorithm fails to be
exact. Better performance can be gained by enabling inexact Grisu3 and
SSE 4.2 in build options, but likely not worthwhile in praxis.

## operation: flatcc json parser and printer for C encode (optimized)

(encode means printing from existing binary buffer to JSON)

    elapsed time: 1.407 (s)
    iterations: 1000000
    size: 722 (bytes)
    bandwidth: 513.068 (MB/s)
    throughput in ops per sec: 710619.931
    throughput in 1M ops per sec: 0.711
    time per op: 1.407 (us)

## operation: flatcc json parser and printer for C decode/traverse (optimized)

(decode/traverse means parsing json to flatbuffer binary and calculating checksum)

    elapsed time: 2.218 (s)
    iterations: 1000000
    size: 722 (bytes)
    bandwidth: 325.448 (MB/s)
    throughput in ops per sec: 450758.672
    throughput in 1M ops per sec: 0.451
    time per op: 2.218 (us)

## JSON parsing and printing on same hardware in Virtual Box Ubuntu

Numbers for Linux included because parsing is significantly faster.

## operation: flatcc json parser and printer for C encode (optimized)

    elapsed time: 1.210 (s)
    iterations: 1000000
    size: 722 (bytes)
    bandwidth: 596.609 (MB/s)
    throughput in ops per sec: 826328.137
    throughput in 1M ops per sec: 0.826
    time per op: 1.210 (us)

## operation: flatcc json parser and printer for C decode/traverse

    elapsed time: 1.772 (s)
    iterations: 1000000
    size: 722 (bytes)
    bandwidth: 407.372 (MB/s)
    throughput in ops per sec: 564227.736
    throughput in 1M ops per sec: 0.564
    time per op: 1.772 (us)

