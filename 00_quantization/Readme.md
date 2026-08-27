# 00, Quantization Foundations

## What Quantization Actually Is

Quantization is the process of representing a continuous or high precision numeric value using a discrete set of values that requires fewer bits to store, and in the context of deep learning it specifically means converting a neural network's weights, activations, gradients, or the key and value cache of a transformer from a high precision floating point format, typically thirty two bit or sixteen bit, into a lower precision format such as eight bit integer, four bit integer, or an eight bit floating point format, while trying to keep the model's output as close as possible to what it would have produced at full precision. The reason this matters at all is entirely about resource constraints rather than mathematical elegance. A weight stored in FP32 occupies four bytes, the same weight stored in FP16 or BF16 occupies two bytes, and the same weight stored in INT4 occupies half a byte, meaning a model quantized from FP16 to INT4 requires roughly one quarter of the memory to store its parameters. This matters in production for four distinct and compounding reasons. First, memory capacity, since a model that does not fit in a GPU's memory cannot be served at all without sharding across multiple devices, and quantization can be the difference between needing one GPU or four. Second, memory bandwidth, since for most LLM inference workloads at small to moderate batch size the bottleneck is not how fast the GPU can multiply numbers but how fast it can move weights from high bandwidth memory into the compute units, so a model with four times fewer bytes to move can, in the best case, run close to four times faster in that memory bound regime. Third, compute throughput, since modern GPU tensor cores can execute far more low precision operations per second than high precision ones, for example INT8 tensor core throughput on recent NVIDIA architectures is roughly double FP16 throughput, and INT4 throughput is roughly double INT8 again, which directly accelerates compute bound workloads such as large batch serving. Fourth, dollar cost, since memory capacity, memory bandwidth, and compute throughput all translate directly into how many requests a single GPU can serve per second, and therefore into the cost per million tokens served, which is usually the number that actually matters to the business.

It is worth being precise about what quantization is not. It is not compression in the lossless sense, meaning the original full precision values generally cannot be perfectly recovered from the quantized representation, so quantization always introduces some amount of numerical error, and the entire discipline of quantization algorithm design, which the earlier step by step list catalogues, is really the discipline of minimizing the damage that this unavoidable error does to the model's actual behavior, rather than eliminating the error entirely, which is not possible once you have committed to using fewer bits.

## Number Formats You Need to Know Before Anything Else

Every quantization technique is defined relative to a source format and a target format, so understanding the handful of number formats in common use is a prerequisite, not an optional detail. FP32, single precision floating point, uses one sign bit, eight exponent bits, and twenty three mantissa bits, giving it a very wide dynamic range and high precision, and is the format neural networks were historically trained in, though it is now rarely used for LLM inference because of its memory and bandwidth cost. FP16, half precision floating point, uses one sign bit, five exponent bits, and ten mantissa bits, halving the memory footprint of FP32 but at the cost of a much narrower exponent range, which historically caused overflow and underflow problems during training that required loss scaling to work around. BF16, brain floating point, also uses sixteen bits total but allocates them as one sign bit, eight exponent bits, and seven mantissa bits, meaning it has the same dynamic range as FP32 but less precision than FP16, and this tradeoff turned out to matter enormously in practice because the dynamic range problem was more damaging than the precision problem, which is why BF16 is now the dominant training and inference format for large models on modern hardware. TF32, used internally by NVIDIA tensor cores, is a nineteen bit format with the exponent range of FP32 and the mantissa of FP16, used transparently during certain matrix multiply operations without the developer needing to explicitly manage it. Moving into genuinely low precision territory, INT8 represents integers using eight bits, either signed in the range negative one hundred twenty eight to one hundred twenty seven or unsigned zero to two hundred fifty five, and unlike the floating point formats above, an integer format has no built in concept of scale, meaning INT8 alone cannot represent a neural network weight directly since weights are typically small fractional numbers, which is precisely why quantization requires the scale and zero point machinery described in the next section. INT4 halves this again to sixteen representable signed levels typically in the range negative eight to seven, and is the most common target for weight only LLM quantization today because it sits at the sweet spot of aggressive compression with generally recoverable accuracy when paired with a good calibration algorithm. FP8 reintroduces a floating exponent at eight total bits, most commonly in two configurations, E4M3 with four exponent bits and three mantissa bits favoring precision over range, and E5M2 with five exponent bits and two mantissa bits favoring range over precision, and because it retains an exponent, FP8 generally handles the outlier heavy distributions found in transformer activations more gracefully than INT8 does at the same bit budget, which is why FP8 has become the preferred low precision format on Hopper and newer GPU generations that support it natively in hardware.

```python
# A quick way to build genuine intuition for why these formats behave
# differently: inspect the actual bit layout and the resulting
# representable range and precision at a given magnitude.

import numpy as np

def describe_dtype(value: float, dtype: np.dtype) -> None:
    """Cast a Python float into a given numpy dtype and report what
    information survives the round trip, which is the core experiment
    for understanding precision loss before any quantization algorithm
    is involved at all."""
    arr = np.array([value], dtype=dtype)
    print(f"{dtype.__name__:>10}: stored={arr[0]!r}  "
          f"error={abs(float(arr[0]) - value):.6e}")

test_value = 0.1234567
for dtype in (np.float32, np.float16):
    describe_dtype(test_value, dtype)

# BF16 is not a native numpy dtype on all platforms; ml_dtypes or torch
# is typically used instead:
#
#   import torch
#   x = torch.tensor(test_value, dtype=torch.bfloat16)
#   print(float(x), abs(float(x) - test_value))
```

## The Core Mathematics: Affine Quantization

Nearly every quantization technique you will encounter, no matter how sophisticated its calibration procedure, ultimately reduces to the same two operations at inference time, and understanding these two operations cold is the single highest leverage thing you can do before studying any specific algorithm. The forward operation, quantization, maps a real valued number x to an integer q using a scale factor s and a zero point z, computed as q equals round of x divided by s, plus z, and then clamped to the representable integer range of the target bit width, for example negative eight to seven for signed INT4. The reverse operation, dequantization, approximately reconstructs the real value from the stored integer by computing x approximately equals s times the quantity q minus z. The scale s is what determines the resolution of the quantization grid, meaning the smallest difference between two real values that can still be distinguished after quantization, and it is derived from the range of real values you need to represent, typically s equals the quantity max minus min of the observed values, divided by the number of representable integer levels. The zero point z is an integer offset that lets the quantization grid represent a range that is not symmetric around zero, which matters because activations after common operations such as ReLU are strictly non negative, and forcing a symmetric grid onto a non negative distribution wastes half of the available integer levels representing negative numbers that never occur. Symmetric quantization is the special case where the real valued range is treated as symmetric around zero, the zero point is fixed at zero, and only the scale needs to be computed, which is simpler, faster to compute with in hardware, and is the standard choice for weight quantization since trained weight distributions are typically already close to symmetric. Asymmetric quantization computes both a scale and a nonzero zero point, and is generally preferred for activation quantization precisely because activation distributions are frequently skewed or strictly one sided.

```python
import torch

def affine_quantize(
    x: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The two-line operation that underlies essentially every quantization
    scheme discussed in this series. Everything more advanced, GPTQ's
    Hessian based column updates, AWQ's activation aware channel scaling,
    SmoothQuant's outlier migration, is a smarter way of choosing what
    to quantize and how to precondition it before this exact operation
    is applied.

    Returns:
        q: the quantized integer tensor
        scale: the scale factor s
        zero_point: the integer zero point z (0 if symmetric)
    """
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax - 1
        max_abs = x.abs().max().clamp_min(1e-8)
        scale = max_abs / qmax
        zero_point = torch.tensor(0)
        q = torch.round(x / scale).clamp(qmin, qmax)
    else:
        qmax = 2 ** bits - 1
        qmin = 0
        x_min, x_max = x.min(), x.max()
        scale = ((x_max - x_min) / qmax).clamp_min(1e-8)
        zero_point = torch.round(-x_min / scale).clamp(qmin, qmax)
        q = torch.round(x / scale + zero_point).clamp(qmin, qmax)

    return q.to(torch.int32), scale, zero_point


def affine_dequantize(q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """The reverse mapping. Note this recovers an approximation of the
    original tensor, not the original tensor itself; the gap between
    the two is the quantization error discussed in the next section."""
    return (q.float() - zero_point.float()) * scale
```

## Quantization Error, Rounding, and Clipping

Two distinct sources of error are introduced by the affine mapping above, and conflating them is a common source of confusion when debugging a quantized model's accuracy. Rounding error comes from the round operation itself, since a real value that falls between two adjacent grid points must be assigned to the nearer one, and the maximum possible rounding error is exactly half of the scale, meaning a smaller scale, which comes from a narrower dynamic range or a higher bit width, always produces smaller rounding error. Clipping error, sometimes called saturation error, comes from values that fall outside the representable range entirely and must be clamped to the minimum or maximum representable integer, and unlike rounding error, clipping error can be arbitrarily large for a single value, which is why a small number of extreme outliers in a tensor can dominate the total error budget even though they represent a tiny fraction of the values. This tension is exactly what drives the entire design space of calibration algorithms, because there is a direct tradeoff between rounding error and clipping error controlled by how wide you set the quantization range, a wider range reduces clipping error on outliers but increases rounding error on the much more numerous typical values by making the scale coarser, and a narrower range does the reverse, which is precisely why naive min max calibration, which sets the range to cover every observed value including rare outliers, is usually suboptimal, and why more sophisticated calibrators use a statistical criterion such as minimizing KL divergence between the original and quantized activation histograms, or clipping to a percentile of the observed distribution, deliberately accepting some clipping error on rare extreme values in exchange for meaningfully better resolution on the bulk of the distribution. This exact tension, and the observation that certain outlier channels in transformer activations are not rare noise but a small, systematic, and functionally important part of the signal, is the underlying reason later techniques such as LLM.int8() and SmoothQuant exist at all, since naive range based calibration handles systematic outliers poorly no matter how the range is tuned, and those techniques instead change what gets quantized or how it is preconditioned before quantization rather than simply tuning the range.

## Why Weights and Activations Behave Differently

A distinction that will recur constantly in every later module is that weights and activations are not equally difficult to quantize, and understanding why up front will make every subsequent algorithm's design choices feel motivated rather than arbitrary. Trained weight matrices tend to have distributions that are roughly bell shaped and centered near zero, with no strong systematic structure tying large magnitude values to a particular position, which makes them relatively forgiving to quantize with a straightforward per channel or per group scheme. Activations, meaning the intermediate outputs flowing through the network at inference time, behave very differently in transformer models specifically, because empirical studies have repeatedly found that a small number of feature dimensions consistently produce activation magnitudes many times larger than the rest of the tensor, and critically, these outlier dimensions are consistent across different input tokens and different examples, meaning they represent a real, structured, and functionally important part of what the model has learned rather than statistical noise that calibration can simply average away. This is precisely why weight only quantization, leaving activations in full precision, is both easier to get right and the default starting point for LLM quantization, while weight and activation quantization requires the additional machinery, outlier isolation, smoothing transformations, or rotations, covered in the later modules of this series.

## Differentiability and the Straight Through Estimator

The round operation at the heart of affine quantization has a derivative of zero almost everywhere and is undefined at the grid points themselves, which means that if you tried to train a network with quantization operations inserted directly into the forward pass, ordinary backpropagation would compute a zero gradient for every parameter that passes through a quantizer, making the parameters impossible to update through gradient descent. The straight through estimator, abbreviated STE, is the practical workaround adopted by every quantization aware training and quantization aware fine tuning technique, and the idea is simple, during the backward pass, treat the round operation as if it were the identity function, meaning the gradient is simply passed through unchanged rather than multiplied by the true, mostly zero, derivative. This is a deliberate approximation that has no rigorous justification in the sense of being an unbiased gradient estimator, but it works remarkably well in practice, and it is the single mechanism that makes QAT, QLoRA style fine tuning through a frozen quantized backbone, and any other technique that needs to backpropagate through a quantization operation, function at all.

## How Quantization Quality Is Measured

Before any specific algorithm is evaluated in later modules, it is important to establish the vocabulary for judging whether a given quantization scheme is good enough. Intrinsic error metrics operate directly on tensors and are cheap to compute, the most common being mean squared error between the original and dequantized tensor, and cosine similarity between the original and dequantized tensor treated as a flattened vector, both of which are useful for quickly comparing two calibration methods on the same layer without running the full model. Perplexity, computed by running the quantized model over a held out text corpus and measuring how well it predicts the next token, is the standard intrinsic metric for judging an entire quantized language model, and is popular because it requires no task specific labels, but it is a necessary rather than sufficient signal, since published results consistently show that perplexity can look nearly unchanged after quantization while specific downstream capabilities, particularly instruction following, multi step reasoning, or tool calling reliability, degrade more than the perplexity number would suggest. Task specific evaluation, meaning running the quantized model through the actual evaluation harness used for the product it will power, whether that is a coding benchmark, a retrieval augmented question answering set, or a human preference comparison, is the metric that should ultimately gate any production decision, and every phase of the execution roadmap discussed earlier in this series explicitly returns to this same task specific evaluation set rather than relying on perplexity alone.

## Summary and What Comes Next in This Series

Quantization is fundamentally a resource versus accuracy tradeoff implemented through the affine mapping of real values to a small set of integers or low precision floats, governed by a scale and an optional zero point, and every sophisticated technique in the field is ultimately a smarter way of choosing that scale and zero point, or of transforming the tensor before the mapping is applied, in order to minimize the combined damage of rounding error and clipping error given the specific, and often outlier heavy, distribution of the tensor being quantized. With this foundation in place, module 01 should move to granularity and calibration in depth, meaning per tensor versus per channel versus per group schemes and the specific calibration criteria such as entropy and percentile clipping referenced above, since that is the next layer of the design space that sits directly on top of the mathematics covered here, before moving into the specific PTQ algorithms, GPTQ, AWQ, SmoothQuant, and the rest of the family cataloged in the earlier step by step list.
