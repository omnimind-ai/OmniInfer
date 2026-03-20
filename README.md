<p align="center">
  <picture>
    <img alt="OmniOp" src="./assets/OmniOp-NPU-logo.png" width="60%">
  </picture>
</p>

<h3 align="center">
High-Performance Operators for Qualcomm Hexagon NPU Acceleration
</h3>

<p align="center">
| 
<a href="https://github.com/omnimind-ai/OmniInfer-LLM"><b>OmniInfer-LLM</b></a> 
| 
<a href="#compilation"><b>Compilation</b></a> 
| 
<a href="#about-fp16-hmx"><b>HMX Technical Details</b></a> 
| 
<a href="https://github.com/omnimind-ai/OmniOp-NPU/issues"><b>Issues</b></a> 
|
</p>

---

This repository provides the operator library required to build and run
[OmniInfer-VLM](https://github.com/omnimind-ai/OmniInfer-VLM) on Qualcomm Hexagon NPU.
It focuses on compiling and integrating NPU-accelerated operators for LLM/VLM
inference on smartphones.

**Hardware requirements**: Android phones with Qualcomm Snapdragon 8 Gen 2 or higher SoC, specifically requiring Hexagon DSP version 73 or above. Note that this implementation relies on **FP16 HMX**, which may not be available on some mid-to-low-end devices.

**Software requirements**: Hexagon SDK 6.x (verified version: 6.0.0.2)

## Compilation

1. First, ensure the Hexagon SDK environment is set up. Run the following command in the root directory of the Hexagon SDK:

```sh
source setup_sdk_env.source
```

2. Execute the following **two** commands in the root directory of this project:

```sh
build_cmake android
build_cmake hexagon DSP_ARCH=v73
```

Here, `DSP_ARCH` specifies the target Hexagon NPU architecture version. We recommend using `v73` by default for better compatibility. (The NPU architecture version on Snapdragon 8 Gen 2 is v73; you can modify this option according to your target hardware.)

After compilation, you should see two directories: `android_ReleaseG_aarch64` and `hexagon_ReleaseG_toolv87_v73` (the actual names may vary depending on the compilation mode, specific toolchain, and target architecture version). Note the following two products:

- `android_ReleaseG_aarch64/libhtp_ops.so`
- `hexagon_ReleaseG_toolv87_v73/libhtp_ops_skel.so`

These two dynamic link libraries will be used later; please refer to the instructions in the main repository. In FastRPC terminology, they are the Stub (`libhtp_ops.so`) and Skeleton (`libhtp_ops_skel.so`) respectively. You can use `ldd` to distinguish between the two shared objects: `libhtp_ops.so` targets the AArch64 architecture and runs on the CPU; `libhtp_ops_skel.so` targets the Q6DSP architecture and runs on the Hexagon NPU (cDSP).

## Known Issues

Compiling for DSP Arch v79 targets with higher versions of the Hexagon SDK may cause errors in some floating-point calculations. When the target hardware is DSP Arch v79, we recommend compiling for v73/v75 and disabling `vgather`-based exp calculations (set `const bool enable_vgather_exp` to `false` in `simple_flash_attn_f16_core` in `src/dsp/ops/flash_attn.c`).

## About FP16 HMX

Our current implementation heavily relies on FP16 HMX for dequantize GEMM and FlashAttention. The relevant instructions are derived from the qhl_hmx sample in Hexagon SDK 5.x (removed in newer versions).

For DSP versions v73 and above, the following inline assembly snippet is used to load tile data (FP16 Croutons) from TCM and accumulate the tile-level inner product results into the internal accumulator. The HMX activation loading instruction and weight loading instruction must be in the same instruction packet.

```C
static HMX_INLINE_ALWAYS void hmx_load_tiles_fp16(const __fp16 *row_tiles, const __fp16 *col_tiles, size_t n_tiles) {
  size_t limit = n_tiles * HMX_FP16_TILE_SIZE - 1;
  asm volatile(
    "{ activation.hf = mxmem(%0, %1):deep\n"
    "weight.hf = mxmem(%2, %3) }\n" ::"r"(row_tiles),
    "r"(limit), "r"(col_tiles), "r"(limit)
    : "memory");
}
```

The following inline assembly snippet is used to output the content of the internal accumulator to TCM. Judging from the name, HMX contains a converter component, but its conversion capabilities are not yet clear.

```C
static HMX_INLINE_ALWAYS void hmx_consume_accumulator_fp16(__fp16 *out) {
  asm volatile(
    "cvt.hf = acc(%0)\n"
    "mxmem(%1, %2) = cvt\n" ::"r"(2),
    "r"(out), "r"(0)
    : "memory");
}
```
The input and output tiles of FP16 HMX here follow the FP16 Crouton layout, which is shown in Figure 4 of our paper. You can also refer to the `test_mat_mul_rpc` function in `src/host/test.c` for details.

The HMX component also supports per-channel scale & bias for the output tile. The size of this data region is 256 bytes.

```C
static HMX_INLINE_ALWAYS void hmx_set_output_scales(const void *scales) {
  asm volatile("bias = mxmem2(%0)" ::"r"(scales));
}
```

**Note that the above information is specific to a particular sequence of HMX instructions. There may be other FP16 HMX instructions that require different data layouts and have functional differences.** There are numerous instruction variants of HMX, and it is difficult for us to associate them with all existing layouts. You can check QNN's `include/QNN/HTP/core/memory_layout.h` and `include/QNN/HTP/core/tile_extract.h` for some information about HMX layouts. If you discover other usable HMX instructions, please feel free to share them with us.

## Contact Us

For technical inquiries regarding NPU operator implementation, assembly optimization, or bug reports, please reach out via our community channels.

<table align="center">
  <tr>
    <td align="center">
      <img src="./assets/wechat_qr.png" alt="WeChat Group" width="220"/><br/>
      <b>WeChat Group</b>
    </td>
    <td align="center">
       <a href="https://github.com/omnimind-ai/OmniOp-NPU/issues">
        <img src="https://img.shields.io/badge/GitHub-Issues-blue?logo=github" alt="GitHub Issues"/>
      </a><br/>
      <b>Technical Support</b>
    </td>
  </tr>
</table>

---