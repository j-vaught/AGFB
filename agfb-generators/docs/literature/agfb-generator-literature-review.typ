#set document(title: "Analytic Gradient Filter Benchmark Generator Literature Review", author: "J.C. Vaught")
#set page(paper: "us-letter", margin: (x: 0.65in, y: 0.7in))
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.56em)

#align(center)[
  #text(size: 17pt, weight: "bold")[Analytic Gradient Filter Benchmark Generator Literature Review]

  #v(0.25em)
  #text(size: 10.5pt)[J.C. Vaught]

  #text(size: 9.5pt)[May 18, 2026]
]

#v(0.7em)

= Purpose

This review records the filter metrics and synthetic image families that are relevant to the Analytic Gradient Filter Benchmark generator set. The reviewed sources cover edge detection, nonlinear scale-space filtering, corner and junction detection, phase congruency, ridge and line detection, vesselness, vascular tree synthesis, and common reference implementations. The main design conclusion is that the generator set should not be limited to isolated smooth Gaussian-like fields. It should include discontinuities, smoothed discontinuities, polynomial fields, roof and line profiles, multi-scale blobs, polygonal junctions, sinusoidal and chirp fields, asymmetric ridges, vessel crossings, vessel bifurcations, and noise or blur transforms with known ground truth.

= Summary

#text(size: 8.4pt)[
  #table(
    columns: (17%, 28%, 28%, 27%),
    inset: 3.5pt,
    stroke: 0.45pt,
    align: left,
    [#strong[Source]],
    [#strong[Metrics]],
    [#strong[Synthetic images or fields]],
    [#strong[Generator target]],
    [@Canny1986EdgeDetection],
    [SNR, localization, one response, duplicate suppression.],
    [Noisy step edges, ridge edges, roof edges.],
    [Sharp steps, blurred steps, roofs, ridges, peaks.],
    [@Perona1990AnisotropicDiffusion],
    [Causality, boundary drift, junction preservation.],
    [Gaussian scale-space, mollified steps, real scale spaces.],
    [Blur and edge-preserving smoothing transforms.],
    [@Harris1988CornerEdge],
    [$R$ response, local maxima, edge-corner-flat class.],
    [Mostly real images, with a clear corner model.],
    [L-corners, T-junctions, checkerboards, controls.],
    [@Kovesi1999PhaseCongruency],
    [Phase congruency, noise threshold, frequency spread.],
    [Steps, square waves, lines, roofs, Mach bands.],
    [Phase-aligned profile families and noisy variants.],
    [@Lindeberg1994JunctionScale],
    [Detection scale, localization scale, junction error.],
    [Diffuse L-junctions, sharp T-junctions, Gaussian noise.],
    [L-, T-, Y-, and X-junction fields.],
    [@Lindeberg1998ScaleSelection],
    [Scale maxima, feature strength, localization, frequency.],
    [Blobs, edges, ridges, junctions, sinusoids, noise.],
    [Multi-scale blobs, ridges, junctions, sine waves, chirps.],
    [@Steger1996Curvilinear],
    [Centerline position, width, bias, bias correction.],
    [Parabolic lines, bars, asymmetric bars, swept curves.],
    [Curved ridges with centerline and width truth.],
    [@Frangi1998Vesselness],
    [Hessian ratios, structure magnitude, vesselness, scale.],
    [Real angiography and magnetic resonance angiography.],
    [Tubes, variable radii, scale sweeps, anti-targets.],
    [@Hannink2014CrossingVesselness],
    [Sensitivity, specificity, accuracy, threshold sweeps.],
    [Real retinal images with crossings and bifurcations.],
    [Crossings, bifurcations, unequal branches.],
    [@Jassi2011VascuSynthSoftware and @Hamarneh2010VascuSynth],
    [Dice, Jaccard, medial-axis Hausdorff, bifurcations, radii, topology.],
    [Volumetric vascular trees with configurable noise.],
    [Two-dimensional vascular trees now, 3D-compatible API later.],
    [@ScikitImageBlobDocs and @ScikitImageFiltersDocs],
    [Reference LoG, DoG, DoH, Frangi, Sato, Meijering, Hessian, Gabor parameters.],
    [Blob, ridge, neurite, tubeness, and oriented-frequency examples.],
    [Generators that map directly to common filter APIs.],
  )
]

= Paper Notes

== Canny. A Computational Approach to Edge Detection

Canny defines edge detection as an optimization problem with three criteria. The detector should have a high probability of finding real edges and a low probability of marking noise, it should localize edge points accurately, and it should return only one response for a single edge. Those criteria translate directly into measurable benchmark outputs. The generator should therefore provide exact edge masks, signed distance fields, and expected response locations so detection rate, localization error, and duplicate response count can be measured without manual inspection @Canny1986EdgeDetection.

The core synthetic stimulus is the noisy step edge. The paper also points beyond simple steps to ridge and roof edge operators. For this project, that means the edge family should include narrow and blurred smoothed steps, finite-width ridges, roof profiles, impulse-like lines, and controlled additive noise. This source is the strongest argument for separating generator truth from operator truth, because the same analytic edge field can be scored under several detector definitions.

== Perona and Malik. Scale-Space and Edge Detection Using Anisotropic Diffusion

Perona and Malik critique Gaussian scale space because coarse-scale boundaries shift and junctions can be destroyed. Their objectives are causality, meaningful coarse-scale boundaries, intraregion smoothing, edge preservation, and preservation of shape and position through smoothing. They use a spatially varying diffusion coefficient driven by image gradients, so this paper is less about a new isolated image primitive and more about transforming a primitive while keeping its semantic boundary valid @Perona1990AnisotropicDiffusion.

The useful synthetic content is the one-dimensional scale-space family, the blurred step edge used for the edge sharpening analysis, and the comparison between isotropic linear diffusion and anisotropic diffusion. The generator set should include paired transforms. One transform should apply standard Gaussian blur to an analytic field, and another should approximate edge-preserving smoothing. Metrics should compare boundary drift, junction preservation, and response stability across smoothing level.

== Harris and Stephens. A Combined Corner and Edge Detector

Harris and Stephens introduce a local auto-correlation matrix and the response $R = op("det")(M) - k op("trace")(M)^2$ to classify corner, edge, and flat regions. The detection procedure selects local maxima of the corner response and also uses response sign to identify edges. The metrics are response separability, local maximum correctness, and edge or corner classification quality @Harris1988CornerEdge.

The paper uses a test image and natural imagery rather than a systematic synthetic image suite. The response model still gives clear generator requirements. We need edge-only controls, flat controls, L-corners, T-junctions, X-junctions, checkerboards, and rotated versions. Each should include exact corner points, edge masks, and a ground-truth class map so corner detectors can be evaluated separately from edge detectors.

== Kovesi. Image Features from Phase Congruency

Kovesi frames feature detection around local phase agreement rather than raw gradient magnitude. The important metrics are phase congruency, noise compensation, frequency spread, and stable behavior across different feature types. The paper also uses hysteresis thresholds in its comparisons, so threshold sensitivity is part of the evaluation surface @Kovesi1999PhaseCongruency.

The synthetic signals are directly useful for the generator set. They include square waves and step edges, triangular or roof profiles, line impulses, Mach-band-like patterns, and noisy step profiles. These are useful because they test filters that should respond to features even when local contrast and local intensity shape differ. The generator set should therefore include phase-aligned primitives and not only derivative-friendly Gaussian fields.

== Lindeberg. Junction Detection with Automatic Selection of Detection Scales and Localization Scales

Lindeberg's 1994 junction paper studies automatic detection scale and localization scale. The measurable outputs are selected scale, localized junction position, and distance from known ground truth. This is a direct fit for analytic generators because the ground-truth junction coordinate and arm geometry can be known exactly @Lindeberg1994JunctionScale.

The synthetic images include diffuse L-junctions built from products of error functions and sharp T-junctions with additive Gaussian noise. This supports a junction generator with parameters for arm count, angle, width, blur, contrast, and noise. The same generator can produce L-, T-, Y-, and X-junctions and can score both point localization and edge-arm continuity.

== Lindeberg. Feature Detection with Automatic Scale Selection

The 1998 scale-selection paper generalizes scale-normalized differential operators across blobs, edges, ridges, junctions, and local frequency estimation. The metrics are selected scale, feature strength, localization error, and stability under scaling. It also distinguishes model patterns from real images, which is exactly the distinction needed in a generator benchmark @Lindeberg1998ScaleSelection.

The synthetic content is broad. It includes model blobs, edges, ridges, junctions, polygon-type junctions with Gaussian noise, and sinusoidal signals used to analyze local frequency and quasi-quadrature behavior. The generator set should therefore add scale families for blobs, ridges, edges, junctions, sine waves, chirps, and perspective-like frequency sweeps.

== Steger. An Unbiased Detector of Curvilinear Structures

Steger focuses on unbiased line and curvilinear structure detection. The measurable outputs are sub-pixel centerline position, line width, and bias under different profile assumptions. This paper is important because a ridge detector can appear successful by visual inspection while still having systematic centerline bias @Steger1996Curvilinear.

The synthetic profiles are parabolic lines, symmetric bars, asymmetric bars, and two-dimensional curved structures generated by sweeping one-dimensional profiles along curves. This implies a ridge generator with explicit centerline, width, curvature, profile type, and asymmetry parameters. The metric set should include centerline distance, width error, and sign of bias.

== Frangi. Multiscale Vessel Enhancement Filtering

Frangi et al. define vesselness from Hessian eigenvalues. The metrics are not classical classification metrics in the paper, but the filter objective is explicit. Vesselness should be high for tubular structures, low for plate-like or blob-like structures, and selected over scale according to the vessel radius @Frangi1998Vesselness.

The paper evaluates real two-dimensional digital subtraction angiography and three-dimensional magnetic resonance angiography rather than synthetic images. For AGFB, it still motivates synthetic tubes and anti-target controls. The generator set should include straight vessels, curved vessels, variable-radius vessels, plate-like structures, blob-like structures, and empty backgrounds, all with exact centerline and radius truth.

== Hannink, Duits, and Bekkers. Crossing-Preserving Multi-Scale Vesselness

Hannink et al. evaluate vessel enhancement at crossings and bifurcations, where ordinary Hessian vesselness can fail because multiple orientations meet at one location. Their metrics are sensitivity, specificity, accuracy, threshold sweeps, and separate analysis in crossing and bifurcation regions @Hannink2014CrossingVesselness.

The source uses real retinal imagery rather than synthetic images, but it identifies the missing synthetic stress cases. The generator set should include X-crossings, Y-bifurcations, unequal-radius branches, shallow crossing angles, and contrast imbalance between branches. These fields should expose both segmentation masks and orientation or branch labels so crossing preservation can be measured, not just visualized.

== VascuSynth. Vascular Tree Synthesis

VascuSynth is a benchmark-oriented generator rather than a detector. It produces three-dimensional vascular image data with ground-truth segmentation, bifurcation locations, branch radii, branch properties, and tree hierarchy. The original article names Dice and Jaccard for segmentation overlap, Hausdorff distance between medial axes, errors in bifurcation locations, errors in branch lengths and radii, and vascular tortuosity measures such as distance metric, inflection count metric, and sum of angles metric. The software release also supports degradations such as Gaussian, uniform, salt-and-pepper, and shadow noise @Hamarneh2010VascuSynth @Jassi2011VascuSynthSoftware.

This source is useful because it treats image generation and truth generation as one product. The immediate AGFB version can adopt that idea in two dimensions by generating vascular trees with branch masks, centerlines, radii, bifurcation nodes, and parent-child topology. The interface should not block a later three-dimensional extension.

== scikit-image Reference Implementations

The scikit-image blob and filter documentation is not a benchmark paper, but it is useful for aligning generator knobs with common filter APIs. The blob examples expose Laplacian-of-Gaussian, difference-of-Gaussians, and determinant-of-Hessian parameters such as scale limits and thresholds. The filter API documents vesselness, neuriteness, tubeness, Hessian, and Gabor filters with scale, orientation, and ridge-polarity parameters @ScikitImageBlobDocs @ScikitImageFiltersDocs.

These references argue for a practical compatibility layer. A user testing a new filter definition should be able to select generators that correspond to common operator families. Blob fields should map to LoG, DoG, and DoH assumptions. Ridge and vessel fields should map to Frangi, Sato, Meijering, and Hessian filters. Oriented sinusoidal packets should map to Gabor-style frequency and orientation tests.

= Recommended Generator Additions

The first addition should be polynomial scalar fields. They are not emphasized by the reviewed detector papers, but they provide a clean analytic baseline for gradient, Hessian, curvature, and derivative correctness. Linear, quadratic, cubic, saddle, and mixed-term fields should be included with closed-form gradients and Hessians.

The second addition should be edge and transition families. This includes narrow smoothed steps, error-function steps, finite ramps, smoothed ramps, roof profiles, Mach-band-like profiles, and paired blur or diffusion transforms. These fields support Canny-style detection metrics and Perona-Malik-style boundary preservation metrics.

The third addition should be junction families. L-, T-, Y-, and X-junctions should be generated from sharp polygons and from smoothed error-function compositions. Each instance should carry junction coordinates, edge masks, branch labels, angles, widths, and selected blur or noise settings.

The fourth addition should be ridge, line, and vessel families. Steger-style bar profiles, parabolic ridges, asymmetric ridges, curved centerlines, tubes, crossings, bifurcations, and small vascular trees should be included. The ground truth should include centerlines, widths or radii, segmentation masks, branch topology, and crossing or bifurcation labels.

The fifth addition should be scale and frequency families. Blobs, anisotropic blobs, sine waves, chirps, Gabor-like packets, and local frequency sweeps should be generated with known scale, orientation, phase, and frequency. These fields make it possible to score scale selection and orientation selection directly.

= References

#bibliography("references.bib", style: "ieee")
