# Frequency Adjustment of Images and Videos

https://user-images.githubusercontent.com/313307/159341116-4d7e840e-e779-4062-9979-bd1fb86ef810.mp4

This package provides the official Julia implementation of the method described in the Eurographics 2021 paper:

*Real-time Frequency Adjustment of Images and Videos*\
Rafael L. Germano, [Manuel M. Oliveira](https://inf.ufrgs.br/~oliveira/), and [Eduardo S. L. Gastal](https://inf.ufrgs.br/~eslgastal/)\
Computer Graphics Forum, Volume 40 (2021), Number 2\
Proceedings of Eurographics 2021, Vienna, Austria\
DOI: [https://doi.org/10.1111/cgf.142612](https://doi.org/10.1111/cgf.142612)\
Open Access PDF: [https://diglib.eg.org/handle/10.1111/cgf142612](https://diglib.eg.org/handle/10.1111/cgf142612)

Please refer to the publication above if you use this software. See the [Bibtex](#Bibtex) section below.

Videos illustrating our real-time frequency adjustment interface can be found at our project webpage at [https://www.inf.ufrgs.br/~eslgastal/RealTimeFrequencyAdjustment/](https://www.inf.ufrgs.br/~eslgastal/RealTimeFrequencyAdjustment/) .

<center>
<a href="https://www.inf.ufrgs.br/">
<img src="https://user-images.githubusercontent.com/15923822/159553600-35477896-8a84-43ec-a25e-1287817b0e6c.png" alt="Instituto de Informática -- UFRGS Logo" width="131"/>
</a>
&nbsp;
<a href="https://www.ufrgs.br/">
<img src="https://user-images.githubusercontent.com/15923822/159553602-fd666b11-51ae-46f0-b127-9bc6aa423522.png" alt="Universidade Federal do Rio Grande do Sul (UFRGS) Logo" width="130"/>
</a>
</center>

## Quick Start

Open the `example.ipynb` notebook.

This package supports parallel processing using threads (start your Julia session with `--threads auto`) or using CUDA (see details below).

## Detailed Usage

The following example shows how this package can be used to adjust the frequencies of an image:

```julia
# Packages necessary for this code
using ColorTypes
using TestImages
using FrequencyAdjustment

# Load barbara image with the TestImages package
s = RGB{Float64}.(testimage("barbara_color"))

# Define wave detection parameters (see Spectral Remapping paper for details,
# at https://www.inf.ufrgs.br/~eslgastal/SpectralRemapping/)
detection_parameters = (;
    R = 4, # Expected image downscaling factor
    σ = 3, # Gaussian window standard deviation, in pixels
)

# Detect high-frequency waves in the image using a non-harmonic decomposition,
# then compute the unwrapped phase, and finally reconstruct the image adjusting
# frequencies by 0.5 (ie, reducing by half the frequency of all waves with
# frequency above 0.4/R cycles per sample)
s_0p5 = adjust(s, 0.5; detection_parameters...)

# Repeat all of the steps above but this time adjust frequencies by 0.1
s_0p1 = adjust(s, 0.1; detection_parameters...)
```

<center>
<a href="https://www.ufrgs.br/">
<img src="https://user-images.githubusercontent.com/15923822/159553596-02287179-58a9-499f-8d6b-7532ac360da4.jpg" alt="Result of frequency adjustment"/>
</a>
</center>

To perform several frequency adjustments of the same image more efficiently, one should pre-compute the wave/frequency detection and phase unwrapping steps using a `FrequencyAdjuster` object, as shown in the following example:

```julia
# Example with pre-computation

# The constructor of the FrequencyAdjuster object performs the wave detection
# step for the signal s and stores the resulting data into F.data[:waves]
F = FrequencyAdjuster(s; detection_parameters...)

# Unwrap the phases to obtain the smooth phase functions (see our paper for
# details) and stores the resulting data into F.data[:u_all]
phaseunwrap!(F)

# Reconstruct the image adjusting frequencies
s_0p5 = adjust(F, 0.5) # Reconstructs the image adjusting frequencies by 0.5
s_0p1 = adjust(F, 0.1) # Reconstructs the image adjusting frequencies by 0.1

# NOTE: The adjust function uses the information in F.data[:waves] and the
# unwrapped phases in F.data[:u_all] to reconstruct the image. If F.data[:u_all]
# is not defined (ie, if the phaseunwrap! function was not executed on F),
# then the adjust function will call phaseunwrap! first and the phases will be
# stored in F.data[:u_all].
```

The `detection_parameters` variable in the examples above is used to define the appropriate **Gaussian window size**, in terms of its standard deviation `σ`, and the **spectral circle** used to determine which frequencies are going to be adjusted. The *radius* of the spectral circle is inversely proportional to the expected image downscaling factor `R`, as described in the Spectral Remapping paper that introduces the non-harmonic wave detection algorithm (see [https://inf.ufrgs.br/~eslgastal/SpectralRemapping/](https://inf.ufrgs.br/~eslgastal/SpectralRemapping/)). Note that the detected waves are already stored in the `FrequencyAdjuster` object and thus it is not necessary to pass the detection parameters to `adjust` when using the `FrequencyAdjuster` object. Furthermore, we note that `adjust` may receive an `Array` of any dimension, however we implement specialized functions for the 2-D and 3-D cases.

The `adjust` function by default uses Principal Component Analysis (PCA) to adjust only the PCA-defined channel that maximizes the color variation direction (see the Spectral Remapping paper for details). To use our multi-channel approach, the `adjust_rgb` function should be used instead, as shown in the example below:

```julia
# Example without pre-computation

# Detect waves of each channel, then compute the unwrapped phases using our RGB 
# optimization, and then reconstruct the image adjusting frequencies by 0.5
s_0p5 = adjust_rgb(s, 0.5; detection_parameters...)

############

# Example with pre-computation

# The FrequencyAdjusterMultichannel constructor creates a tuple with one
# FrequencyAdjuster object for each channel (currently only 3-channel images
# are supported)
F = FrequencyAdjusterMultichannel(s; detection_parameters...)

# Unwrap the phases of the waves of all channels simultaneously using our
# RGB optimization to obtain the phases of each channel, and store the resulting
# data into F[i].data[:u_all], for i ∈ (1, 2, 3)
phaseunwrap_rgb!(F)

# Reconstruct the waves in each channel adjusting frequencies
s_0p5 = adjust_rgb(F, 0.5) # Reconstructs the image adjusting frequencies by 0.5
s_0p1 = adjust_rgb(F, 0.1) # Reconstructs the image adjusting frequencies by 0.1

# NOTE: If F[i].data[:u_all] is not defined for any i ∈ (1, 2, 3), then
# the function adjust_rgb will call the phaseunwrap_rgb! function to unwrap
# the phases with our RGB optimization and store the unwrapped phases into F[i].data[:u_all]
```

In this package we also include the implementation of the **anisotropic** frequency adjustment method described in the Thesis available at the [UFRGS LUME digital repository](https://lume.ufrgs.br/handle/10183/225714). To perform anisotropic frequency adjustment use the `adjust` and `adjust_rgb` methods with a `Tuple` of adjustment factors as shown in the example below:

```julia
# Adjust vertical and horizontal frequencies of s by 0.1 and 0.5 respectively
s_0p1_0p5 = adjust(s, (0.1, 0.5); detection_parameters...)

# Adjust vertical and horizontal frequencies of the waves detected in F by 0.1 and 0.5 respectively, using RGB optimization
s_0p1_0p5 = adjust_rgb(F, (0.1, 0.5))

# NOTE: If the FrequencyAdjuster or FrequencyAdjusterMultichannel object passed
# does not contain the data necessary to perform anisotropic adjustment then 
# anisotropic_phaseunwrap! or anisotropic_phaseunwrap_rgb! is called before adjusting the image
```

It is also possible to use a GPU with CUDA support to adjust the image content. To use the GPU it is necessary to first have the *CUDA.jl* package installed and loaded, and then call the `cu_adjust` or `cu_adjust_rgb` methods, or the `adjust` and `adjust_rgb` methods with a `cuFrequencyAdjuster` object, as in the example below:

```julia
using CUDA

# cu_adjust works with the same arguments as adjust, but reconstructs in the device (GPU)
# cu_adjust may be used with
# Arrays (ie, images)
s_0p5 = cu_adjust(s, 0.5; detection_parameters...)
# FrequencyAdjuster objects
F = FrequencyAdjuster(s; detection_parameters...)
s_0p5 = cu_adjust(F, 0.5)
# and cuFrequencyAdjuster objects
cu_F = cuFrequencyAdjuster(F)
d_s_0p5 = cu_adjust(cu_F, 0.5)

# NOTE: the returned Array is a cuArray (thus we prepend a 'd_' to indicate it is stored in the device)

# Alternatively, if one calls the adjust function with a cuFrequencyAdjuster object
# as a parameter, it will behave like cu_adjust (ie, image reconstruction will be
# done in the GPU device)
d_s_0p5 = adjust(cu_F, 0.5)

# cu_adjust supports anisotropic reconstruction as well, however the cuFrequencyAdjuster object must be
# created passing true to the aniso parameter in its constructor
cu_F = cuFrequencyAdjuster(F; aniso=true)
# Then cu_adjust can be called with a tuple of vertical and horizontal frequencies
d_s_0p1_0p5 = cu_adjust(cu_F, (0.1, 0.5))

# For multi-channel reconstruction use cu_adjust_rgb
s_0p5 = cu_adjust_rgb(s, 0.5; detection_parameters...)
```

## Compatibility

- FrequencyAdjustment.jl uses functions of Julia v1.6 and there is no plan to provide support for Julia < v1.6.
- FrequencyAdjustment.jl `cu_*` methods and objects require CUDA.jl.

## Bibtex

Rafael L. Germano, Manuel M. Oliveira and Eduardo S. L. Gastal. "Real-Time Frequency Adjustment of Images and Videos". Computer Graphics Forum. Volume 40 (2021), Number 2, Proceedings of Eurographics 2021. 

```bibtex
@article{GermanoOliveiraGastal2021,
  author   = {Rafael L. Germano and Manuel M. Oliveira and Eduardo S. L. Gastal},
  title    = {Real-Time Frequency Adjustment of Images and Videos},
  journal  = {Computer Graphics Forum},
  volume   = {40},
  number   = {2},
  pages    = {23-37},
  doi      = {https://doi.org/10.1111/cgf.142612},
  url      = {https://www.inf.ufrgs.br/~eslgastal/RealTimeFrequencyAdjustment/},
  eprint   = {https://diglib.eg.org/handle/10.1111/cgf142612},
  year     = {2021}
}
```

## Keywords

Frequency Adjustment, Spectral Remapping, Image Downscaling, Antialiasing, Resampling, Signal Processing, Gabor Transform, Fourier Transform. 
