# Wasserstein Dictionary Learning [Schmitz et al. 2018]

This repository contains the code for the following publication. Please credit this reference if you use it.

    @article{schmitz_wasserstein_2018,
        title = {Wasserstein {Dictionary} {Learning}: {Optimal} {Transport}-based unsupervised non-linear dictionary learning},
        shorttitle = {Wasserstein {Dictionary} {Learning}},
        url = {https://hal.archives-ouvertes.fr/hal-01717943},
        journal = {SIAM Journal on Imaging Sciences},
        author = {Schmitz, Morgan A and Heitz, Matthieu and Bonneel, Nicolas and Ngolè Mboula, Fred Maurice and Coeurjolly, David and Cuturi, Marco and Peyré, Gabriel and Starck, Jean-Luc},
        year = {2018},
        keywords = {Dictionary Learning, Optimal Transport, Wasserstein barycenter},
    }

The full text is available on [HAL](https://hal.archives-ouvertes.fr/hal-01717943) and [arXiv](https://arxiv.org/abs/1708.01955).


### Configure, build and run

There is a CMakeLists.txt for the project, so you can just create a build directory *outside the source*.

    $ mkdir build
    $ cd build
    $ ccmake ../inverseWasserstein/cpp/

In Cmake, you can activate the different options and targets, then configure and generate a project file for your system (Makefile, VS project, etc.).

Target :
- `BUILD_APP_DICTIONARY_LEARNING` : Target for Dictionary Learning on images (2D histograms)

The different options allow to link different libraries in order to use different kernels (see below)
- `WITH_HALIDE`
- `WITH_EIGEN`
- `WITH_AVX_SUPPORT`
- `WITH_OPENMP`

###  Kernels

The kernel is the part of the algorithm that does the convolution in the Sinkhorn barycenter algorithm. We provide multiple kernels in `kernels.h`.
Some of them use external libraries that are activated with the options mentionned above.
You can change the kernel used, by changing the `typedef XXX KernelType` in `main_dictionary_learning.cpp`.

### Histogram IO

The files histogramIO.h and histogramIO.cpp contain helper functions to load many kinds of histograms (1D,2D,3D,nD). They also contain a bunch of classes to export histograms (in different formats), that are used during the optimization.
The example given only uses the functions for 2D histograms, but you can easily adapt those examples to other types of histograms thanks to these helper functions.

### Halide

[Halide](http://halide-lang.org/) is a language for image processing that uses JIT compilation, and that allows fast convolutions (~3x faster than with SSE). It is optionnal, but we recommend its use. You will need to set the CMake variable HALIDE_ROOT_DIR to the Halide folder you downloaded, in order to use it.

### Files

The 4 important files of the algorithm are :
`inverseWasserstein.h`: Core of the algorithm
`kernels.h`: Classes that compute convolutions in different ways for the Sinkhorn barycenter algorithm.
`loss.h`: Loss functions and their gradients
`histogramIO.h`: Classes and functions for reading and writing histograms of different dimensions, in different format.

The file `main_dictionary_learning.cpp` is an example of how to use the class `inverseWasserstein.h::WassersteinRegression` for your application.

The other files are :
`chrono.h`: Measure execution time
`signArray.h`: Needed if you are using log-domain kernels (see Extensions).
`sse_helpers`: Needed if you are using SSE-based kernels.


### Examples

For the `app_dictionary_learning` program, here is a simple example that runs the algorithm on some toy images.

- Cardiac cycle : (~3 hours on a 16 core)
`./app_dictionary_learning -i ../data/imgheart2 -o outputFiles -k 4 -l 2 -n 25 -s 100 -g 2 -x 200`

- Wasserstein faces : ( ~15 hours on a 16 core)
`./app_dictionary_learning -i ../data/mug_001_expr2 -o outputFiles -k 5 -l 4 -n 100 -s 100 -g 1 -a 3 -x 500 -m 100 --imComplement`

- Wasserstein faces with warm restart : (~2.5 hour on a 16 core)
`./app_dictionary_learning -i ../data/mug_001_expr2 -o outputFiles -k 5 -l 4 -n 5 -s 100 -g 1 -a 3 -x 500 -m 100 --imComplement --warmRestart`


### CLI parameters and options

CLI parameters for `app_dictionary_learning` :

Parameter / Option        | Explanation                                            | Default | Typical
--------------------------|--------------------------------------------------------|---------|---------------
`-i <inDir>`              | Input directory where to find the input histograms     | -       | -
`-o <outDir>`             | Output directory where to write results                | -       | -
`[-k <numOfAtoms>]`       | Number of atoms to find                                | 4       | [2-10]
`[-l <lossType>]`         | Loss function                                          | 2       | [1,4]
`[-n <nIterSinkhorn>]`    | Number of Sinkhorn iterations                          | 25      | [20-500],[2,20]<sup>1</sup>
`[-s <scaleDictFactor>]`  | Factor for scaling between weights and atom values     | 100     | [10,1000]
`[-g <gamma>]`            | Entropic regularization parameter                      | 2       | [0.5,50]
`[-a <alpha>]`            | Value for gamma correction of input images             | 1       | [1-5]
`[-x <maxOptimIter>]`     | Max number of optimization iterations                  | 200     | [50,1000]
`[-m <exportEveryMIter>]` | Saving frequency for intermediate results saving       | 0       | [0,maxOptimIter]
`[--deterministic]`       | Generate random initial weights in a deterministic way | OFF     | -
`[--imComplement]`        | Invert the values of input images                      | OFF     | -
`[--allowNegWeights]`     | Do not force the barycentric weights to be positive.   | OFF     | -
`[--warmRestart]`         | Activate the warm restart technique                    | OFF     | -

<sup>1</sup> Range when using the warm restart

##### Additional information on parameters

- `[-l <lossType>]` : 1 (Total Variation), 2 (Quadratic Loss), 3 (Wasserstein Loss, 4 (Kullback-Leibler Loss)
- `[-s <scaleDictFactor>]` : Should be tuned : a too high value will update the weights too much, and a too low value will update them too little. The values we used were between 10 and 1000, depending on the problem size.
- `[-a <alpha>]` : This is useful when working with 2D images. If there is a close-to-zero background level,
Setting the image to the power $\alpha$ will stretch the range so that values close to zero get even closer to it. This prevents the phenomenon that gathers residual background mass, resulting in a significant amount, and gives the impression that mass is created instead of transported.
- `[--imComplement]` : This is useful to consider either black or white as the presence of mass. Without this option, white is considered as presence of mass.
- `[-m <exportEveryMIter>]`: Set to 0 to save only the final results.
- `[-x <maxOptimIter>]` : Always represent the total number of iterations, even in the warm restart mode.


### Extensions

##### Log-domain stabilization

The log-domain stabilization can be used to use arbitrarily small values of the regularization parameter $\gamma$.
In order to use it, you need to uncomment the line `#define COMPUTE_BARYCENTER_LOG` at the top of the file `main_dictionary_learning.cpp`.

##### Warm restart

The idea is that instead of a single L-BFGS run of 500 iterations, you restart a fresh L-BFGS every 10 iterations, and initialize the scaling vectors as the ones obtained at the end of the previous run.
As explained in the paper, this technique accumulates the Sinkhorn iterations as we accumulate L-BFGS runs, so it allows to compute less Sinkhorn iterations for equivalent or better results, which leads to significant speed-ups.
Be aware that accumulating too much Sinkhorn iterations can lead to numerical instabilities. If that happens, you can use the log-domain stabilization, which is slower, but compensated by the speed-up of the warm restart.
For more details, please refer to our paper.
The value of 10 optimization iterations per L-BFGS run is arbitrary and can be changed in the code (in `regress_both()` of `inverseWasserstein.h`), but it has shown good results for our experiments.



### License

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


### Contact

matthieu.heitz@univ-lyon1.fr
