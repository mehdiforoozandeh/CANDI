---
type: wiki
title: Query decoders and conditional computation
summary: Generating or selecting decoder weights per query instead of sharing one kernel — sparsely-gated mixture-of-experts, CondConv, and dynamic convolution.
category: comparison
sources: raw/shazeer-2017-moe.pdf, raw/yang-2019-condconv.pdf, raw/chen-2019-dynconv.pdf
created: 2026-07-31T23:27:28
updated: 2026-07-31T23:27:28
---

# Query decoders and conditional computation

All three break the same assumption — that one set of weights serves every input — and they differ mainly in whether the mixing happens in output space or weight space, and whether the gate is sparse.

## Sparsely-gated mixture of experts

`raw/shazeer-2017-moe.pdf` is the origin of modern conditional computation. The premise: a network's capacity to absorb information is bounded by its parameter count, and **conditional computation** — activating only parts of the network per example — decouples capacity from per-example compute. The MoE layer holds thousands of expert sub-networks and a trainable gating network that selects a **sparse** combination for each example.

The costs are the paper's real contribution to practice: sparse discrete routing needs a **load-balancing** auxiliary loss to stop the gate collapsing onto a few experts, and it introduces batching and expert-utilisation problems that do not exist in dense layers.

## Mixing in weight space

`raw/yang-2019-condconv.pdf` (CondConv) challenges the assumption that convolutional kernels should be shared across all examples. Instead of mixing expert **outputs**, it computes an input-dependent **convex combination of kernel weights** and then performs a single convolution with the resulting kernel:

    W(x) = Σ_k α_k(x) · W_k ,   y = W(x) * x

This is mathematically distinct from output mixing and much cheaper: one convolution regardless of the number of experts, rather than one per expert. Capacity grows with the number of kernels in the basis while inference cost stays essentially fixed.

`raw/chen-2019-dynconv.pdf` (Dynamic Convolution) arrives at the same construction from the efficiency direction — aggregating multiple parallel kernels by input-dependent **attention** — and emphasises that assembling kernels is cheap because kernels are small, while the aggregation is **non-linear in the input** and therefore adds real representational power. Reported gains: +2.9% top-1 on ImageNet for MobileNetV3-Small at +4% FLOPs.

The two papers are near-simultaneous and describe the same mechanism; CondConv frames it as conditional parameterisation, Dynamic Convolution as attention over kernels.

## Choosing among them

| | routing | mixing space | cost with E experts | needs load balancing |
|---|---|---|---|---|
| Sparse MoE | top-k, discrete | outputs | ~k experts' compute | yes |
| CondConv / DynConv | softmax or sigmoid, dense | weights | one conv + weight blend | no |

For a decoder that must emit an output **per query** (per assay, per target), the weight-space variants are the natural fit: the query supplies the routing signal, the number of queries is small and fixed, and dense softmax routing over a handful of kernels avoids the routing instability that dominates small-scale sparse MoE. The sparse-MoE machinery earns its complexity only when the expert count is large enough that dense mixing is unaffordable.

A caveat both weight-space papers share: because the effective kernel is a per-example convex combination, the layer is **linear in the basis** — expressiveness comes from the input-dependence of α, so a gate that saturates or collapses reduces the layer to an ordinary convolution.

## See also

Related:: [[film-conditioning]], [[set-conditioned-modelling-and-missingness]], [[transformers-and-positional-encoding]], [[covariate-conditioning-and-counterfactuals]]
