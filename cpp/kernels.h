#pragma once
#include <algorithm>

#ifdef HAS_OPENMP
#include <omp.h>
#endif

#include <cmath>

#include <assert.h>
#include <cstring>
#include <complex>


#define SSE_SUPPORT
// #define AVX_SUPPORT

#ifdef HAS_HALIDE
#include "Halide.h"
#endif

#ifdef HAS_EIGEN
#include "Eigen/Eigen"
#endif

#include "sse_helpers.h"
#include "signArray.h"

// Epsilon for log(max(EPSILON, x)) and max(EPSILON, Gaussian(x))
#define EPSILON 1E-250

// Generic
class MatrixKernel;  // General matrix, with values actually stored. For non-symmetric distances
class SymmetricMatrixKernel; // Symmetric matrix, with values actually stored. For symmetric distances
class SymmetricMatrixHalideKernel; // Symmetric matrix, uses Halide for the matrix multiply
class SymmetricMatrixEigenKernel; // Symmetric matrix, uses Eigen for the matrix multiply

// 1D Gaussian convolution
class Gaussian1DKernel;
class GaussianSSE1DKernel;

// 2D Gaussian convolution
class Gaussian2DKernel;   // 2D Gaussian Kernel, for 2-Wasserstein.
class GaussianSSE2DKernel; // same, vectorized with SSE/AVX  (4x speedup ; 1E-14% error)
class GaussianHalide2DKernel; // same, optimized using Halide (about 3x speedup w.r.t SSE)
class LogComplexGaussian2DKernel;
class LogGaussian2DKernel;
class LogSignArrayGaussian2DKernel;
template<typename T>
class LogSignArrayGaussianHalide2DKernel;

// 3D Gaussian convolution (same suffixes)
class Gaussian3DKernel;
class GaussianSSE3DKernel;
class GaussianHalide3DKernel;



#ifndef M_PI
#define M_PI 3.1415926535897932385626
#endif

template<typename T> T sqr(const T x) { return x*x; };

template<typename T>
void matVecMul(const T* M, const T* u, T *v, size_t N) {
#pragma omp parallel for
	for (int i=0; i<N; i++) {
		T r = 0.;
		for (int j=0; j<N; j++) {
			r+= M[i*N+j]*u[j];
		}
		v[i] = r;
	}
}

template<typename T>
void matTransVecMul(const T* M, const T* u, T *v, size_t N) {
#pragma omp parallel for
	for (int i=0; i<N; i++) {
		T r = 0.;
		for (int j=0; j<N; j++) {
			r+= M[j*N+i]*u[j]; // beurk.
		}
		v[i] = r;
	}
}
class MatrixKernel {
public:
	MatrixKernel(double* mat = NULL, int nbcols = 0) {
		matrix = mat;
		N = nbcols;
	}

	void convolve(const double* u, double* result, int nvectors = 1) const {
		for (int i=0; i<nvectors; i++) // sh/could be replaced with dgemm
			matVecMul(matrix, u+i*N, result+i*N, N);
	}
	void convolveAdjoint(const double* u, double* result, int nvectors = 1) const {
		for (int i=0; i<nvectors; i++) // could be replaced with dgemm
			matTransVecMul(matrix, u+i*N, result+i*N, N);
	}

	double operator()(int i, int j) {
		return matrix[i*N+j];
	}
	size_t N;
	double* matrix;
};

class SymmetricMatrixKernel {
public:
	SymmetricMatrixKernel(double gamma = 1., double* mat = NULL, int nbcols = 0) {
		matrix = mat;
		N = nbcols;
		this->gamma = gamma;
	}

	void convolve(const double* u, double* result, int nvectors = 1) const {
		for (int i=0; i<nvectors; i++) // sh/could be replaced with dgemm
			matVecMul(matrix, u+i*N, result+i*N, N);
	}
	void convolveAdjoint(const double* u, double* result, int nvectors = 1) const {
		convolve(u, result, nvectors);
	}

	double operator()(int i, int j) {
		return matrix[i*N+j];
	}
	size_t N;
	double* matrix;
	double gamma;
};


#ifdef HAS_HALIDE
using namespace Halide;
typedef Buffer<double> Bufferd;

static inline Func matrixVec_Halide(ImageParam &A, ImageParam &x) {

	Var i("i"), j("j");
	Func result("result");
	int nbthreads = 2;// omp_get_max_threads();

	const Expr size = A.width();

	const int block = 1;
	//const Expr lastblock = (size/block)*block;
	//RDom kmain(0, lastblock, "kmain");
	//RDom ktail(lastblock, size-lastblock, "ktail");
	RDom k(0, size, "k");
	result(i)  = cast<double>(0);
	result(i) += A(k, i) * x(k);
	//result(i) += A(ktail, i) * x(ktail);
	//result(i) = Halide::sum(A(k, i)*x(k));

	Var xi("xi"), yi("yi"), xo, yo, tile;

	//result.update().allow_race_conditions();
	result.update().parallel(i); //0.0285

	A.set_min(0, 0).set_min(1, 0);
	x.set_bounds(0, 0, A.height());

	return result;
}


static inline Func transpose(ImageParam im) {
	Func transpose_tmp("transpose_tmp"), im_t("im_t");
	Var i("i"), j("j"), ii("ii"), ji("ji"),
		ti("ti"), tj("tj"), t("t");

	transpose_tmp(i, j) = im(j, i);
	im_t(i, j) = transpose_tmp(i, j);

	Expr rows = im.width(), cols = im.height();

	im_t.compute_root()
		.specialize(rows >= 4 && cols >= 4)
		.tile(i, j, ii, ji, 4, 4).vectorize(ii).unroll(ji)
		.specialize(rows >= 128 && cols >= 128)
		.tile(i, j, ti, tj, i, j, 16, 16)
		.fuse(ti, tj, t).parallel(t);

	transpose_tmp.compute_at(im_t, i)
		.specialize(rows >= 4 && cols >= 4).vectorize(j).unroll(i);

	return im_t;
}


static inline Func matrixMat_Halide(ImageParam &A, ImageParam &B) {

	const int vec_size = 4;

	Var i("i"), j("j");
	Var ii("ii"), ji("ji");
	Var ti[4], tj[4], t;
	Func result("result");

	const Expr num_rows = A.width();
	const Expr num_cols = B.height();
	const Expr sum_size = A.height();

	const Expr sum_size_vec = sum_size / vec_size;

	// Pretranspose A and/or B as necessary

	Var k("k");
	Func prod;
	// Express all the products we need to do a matrix multiply as a 3D Func.
	prod(k, i, j) = A(k, i) * B(k, j);

	// Reduce the products along k using whole vectors.
	Func dot_vecs("dot_vecs");
	RDom rv(0, sum_size_vec);
	dot_vecs(k, i, j) += prod(rv * vec_size + k, i, j);

	// Transpose the result to make summing the lanes vectorizable
	Func dot_vecs_transpose("dot_vecs_transpose");
	dot_vecs_transpose(i, j, k) = dot_vecs(k, i, j);

	Func sum_lanes("sum_lanes");
	RDom lanes(0, vec_size);
	sum_lanes(i, j) += dot_vecs_transpose(i, j, lanes);

	// Add up any leftover elements when the sum size is not a
	// multiple of the vector size.
	Func sum_tail("sum_tail");
	RDom tail(sum_size_vec * vec_size, sum_size - sum_size_vec * vec_size);
	sum_tail(i, j) += prod(tail, i, j);

	// Add the two.
	Func AB("AB");
	AB(i, j) = sum_lanes(i, j) + sum_tail(i, j);

	// Do the part that makes it a 'general' matrix multiply.
	result(i, j) = AB(i, j);

	// There's a mild benefit in specializing the case with no
	// tail (the sum size is a whole number of vectors).  We do a
	// z-order traversal of each block expressed using nested
	// tiling.

	/*result
	.specialize(num_rows >= 16*num_cols && num_cols >= 8)
	.tile(i, j, ii, ji, 64, 4).vectorize(ii).unroll(ji)
	.tile(i, j, ti[0], tj[0], i, j, 2, 2)
	.specialize(num_cols >= 16)
	.tile(ti[0], tj[0], ti[1], tj[1], 2, 2)
	.specialize(num_cols >= 32)
	.tile(ti[0], tj[0], ti[2], tj[2], 2, 2)
	.specialize(num_cols >= 64)
	.tile(ti[0], tj[0], ti[3], tj[3], 2, 2);

	result.specialize(num_rows >= 16*num_cols && num_cols >= 8).specialize(num_rows>=256)
	.fuse(tj[0], ti[0], t).parallel(t);*/

	result
		.specialize(sum_size == (sum_size / 8) * 8)
		.specialize(num_rows >= 4 && num_cols >= 2)
		.tile(i, j, ii, ji, 4, 2).vectorize(ii).unroll(ji)
		.specialize(num_rows >= 8 && num_cols >= 8)
		.tile(i, j, ti[0], tj[0], i, j, 2, 4)
		.specialize(num_rows >= 16 && num_cols >= 16)
		.tile(ti[0], tj[0], ti[1], tj[1], 2, 2)
		.specialize(num_rows >= 32 && num_cols >= 32)
		.tile(ti[0], tj[0], ti[2], tj[2], 2, 2)
		.specialize(num_rows >= 64 && num_cols >= 64)
		.tile(ti[0], tj[0], ti[3], tj[3], 2, 2)
		.specialize(num_rows >= 128 && num_cols >= 128)
		.fuse(tj[0], ti[0], t).parallel(t);


	// The general case with a tail (sum_size is not a multiple of
	// vec_size). The same z-order traversal of blocks of the
	// output.
	result
		.specialize(num_rows >= 4 && num_cols >= 2)
		.tile(i, j, ii, ji, 4, 2).vectorize(ii).unroll(ji)
		.specialize(num_rows >= 8 && num_cols >= 8)
		.tile(i, j, ti[0], tj[0], i, j, 2, 4)
		.specialize(num_rows >= 16 && num_cols >= 16)
		.tile(ti[0], tj[0], ti[1], tj[1], 2, 2)
		.specialize(num_rows >= 32 && num_cols >= 32)
		.tile(ti[0], tj[0], ti[2], tj[2], 2, 2)
		.specialize(num_rows >= 64 && num_cols >= 64)
		.tile(ti[0], tj[0], ti[3], tj[3], 2, 2)
		.specialize(num_rows >= 128 && num_cols >= 128)
		.fuse(tj[0], ti[0], t).parallel(t);


	dot_vecs
		.compute_at(result, i).unroll(i).unroll(j)
		.update().reorder(i, j, rv).unroll(i).unroll(j);
	dot_vecs_transpose
		.compute_at(result, i).unroll(i).unroll(j);
	sum_lanes
		.compute_at(result, i).update().unroll(lanes);
	sum_tail
		.compute_at(result, i)
		.update().reorder(i, j, tail).unroll(i).unroll(j);


	dot_vecs.vectorize(k).update().vectorize(k);
	dot_vecs_transpose.vectorize(k);

	// The following stages are only vectorizable when we're
	// computing multiple dot products unrolled.
	Expr can_vectorize = num_rows >= 4 && num_cols >= 2;
	sum_tail.specialize(can_vectorize).fuse(i, j, t).vectorize(t);
	sum_lanes.specialize(can_vectorize).fuse(i, j, t).vectorize(t);
	sum_lanes.update().specialize(can_vectorize).fuse(i, j, t).vectorize(t);


	A.set_min(0, 0).set_min(1, 0);
	B.set_bounds(0, 0, sum_size).set_min(1, 0);

	result.output_buffer().set_bounds(0, 0, num_rows).set_bounds(1, 0, num_cols);

	return result;
}

static inline Func my_matmat_from_matvec_Halide(ImageParam &A, ImageParam &B) {

	Var x("x"), y("y"), z("z");
	Func result("result");

	Expr A_cols = A.width();
	Expr A_rows = A.height();
	Expr B_cols = B.width();
	Expr B_rows = B.height();

//	A_cols = print(A_cols, " ", A_rows, " ", B_cols," ", B_rows);

	// Create a 3D array of products
	// B is contiguous in memory along the first coordinate.
	// Its shape is the transposed one to do the product.
	// We don't need to transpose it, we just need to read the data correclty.
	// A	N
	// (----------)
	// (----------) N
	// (----------)
	// (----------)
	// B	N
	// (----------)
	// (----------) M
	// (----------)
	Func prod("prod");
	prod(x, y, z) = A(z, x) * B(z, y);
//	prod(x, y, z) = print(A(z, x), "A(",z,x,")") * print(B(z, y), "B(",z,y,")");

	// Doesn't work... B values are 0, idk why
//	RDom k(0, A_cols, "k");
//	result(x,y) = 0;
//	result(x,y) += prod(x, y, k);

	// Reduction : Sum along z
	RDom k(0, A_cols, "k");
	result(x,y) = sum(prod(x, y, k));
//	result(x,y) = sum(A(k, x) * B(k, y));

//	prod.trace_stores();
//	prod.trace_loads();
//	result.trace_stores();
//	result.print_loop_nest();

	Var xo,xi;
	// I want to vectorize the computation of prod along z, but it doesn't seem to work...
//	prod.vectorize(z,4); // Cannot vectorize dimension z.v27 of function prod because the function is scheduled inline
//	result.vectorize(k,4); // In schedule for result$1, could not find split dimension: k$x
//	prod.compute_at(result,x); // Func "prod" is computed at the following invalid location (Legal locations for this function are:...)
	result.split(x,xo,xi,32).parallel(xo); // 3.7
//	result.split(x,xo,xi,128).parallel(xo); // 3.9
//	result.split(x,xo,xi,16).parallel(xo); // 3.7
//	result.split(x,xo,xi,A_cols/32).parallel(xo); // 4
	return result;
}


class SymmetricMatrixHalideKernel {
public:
	SymmetricMatrixHalideKernel(double regularization = 1., double* mat = NULL, size_t nbcols = 0) {
		gamma = regularization;
		N = nbcols;
		matrix = mat;
		m = new ImageParam(type_of<double>(), 2, "m");
		v = new ImageParam(type_of<double>(), 1, "v");
		vm = new ImageParam(type_of<double>(), 2, "vm");
		matvec = matrixVec_Halide(*m, *v);

//		matmat = matrixMat_Halide(*m, *vm);
		matmat = my_matmat_from_matvec_Halide(*m,*vm);

		Target target = get_host_target();
		matvec.compile_jit(target);
		matmat.compile_jit(target);
	}
	SymmetricMatrixHalideKernel(const SymmetricMatrixHalideKernel& b) {
		gamma = b.gamma;
		m = b.m; //beware, copies pointers
		v = b.v; //beware, copies pointers
		vm = b.vm; //beware, copies pointers
		matrix = b.matrix; //beware, copies pointers
		N = b.N;
		matvec = b.matvec;
		matmat = b.matmat;
	}
	SymmetricMatrixHalideKernel& operator=(const SymmetricMatrixHalideKernel& b) {
		gamma = b.gamma;
		m = b.m; //beware, copies pointers
		v = b.v; //beware, copies pointers
		vm = b.vm; //beware, copies pointers
		matrix = b.matrix; //beware, copies pointers
		N = b.N;
		matvec = b.matvec;
		matmat = b.matmat;
		return *this;
	}
	~SymmetricMatrixHalideKernel() {

	}
	double operator()(int i, int j) {
		return matrix[i*N+j];
	}
	void convolve(const double* u, double* result, int nvectors = 1) {

		if (nvectors==1) {
			Bufferd resbuff(result, N);
			Bufferd matbuff(matrix, N, N);
			Bufferd vbuff(const_cast<double*>(u), N);
			m->set(matbuff);
			v->set(vbuff);
			matvec.realize(resbuff);
		} else {
			Bufferd resbuff(result, N, nvectors);
			Bufferd matbuff(matrix, N, N);
			Bufferd vbuff(const_cast<double*>(u), N, nvectors);
			m->set(matbuff);
			vm->set(vbuff);
			matmat.realize(resbuff);
		}
	}

	void convolveAdjoint(const double* u, double* result, int nvectors = 1) {
		convolve(u, result, nvectors);
	}

	double gamma;
	size_t N;
	ImageParam *m;
	ImageParam *v;
	ImageParam *vm;
	double* matrix;
	Func matvec;
	Func matmat;
};

#endif



#ifdef HAS_EIGEN

class SymmetricMatrixEigenKernel {
public:
	SymmetricMatrixEigenKernel(double gamma = 1., double* mat = NULL, int nbcols = 0) {
		matrix = mat;
		N = nbcols;
		this->gamma = gamma;
		Eigen::initParallel();
	}

	void convolve(const double* u, double* result, int nvectors = 1) const {
		//for (int i=0; i<nvectors; i++) // sh/could be replaced with dgemm
		//	matVecMul(matrix, u+i*N, result+i*N, N);

		Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > matConv(matrix, N, N);
		Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > matRhs(u, N, nvectors);
		Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > matResult(result, N, nvectors);
		matResult = matConv*matRhs;

	}
	void convolveAdjoint(const double* u, double* result, int nvectors = 1) const {
		convolve(u, result, nvectors);
	}

	double operator()(int i, int j) {
		return matrix[i*N+j];
	}
	size_t N;
	double* matrix;
	double gamma;
};

#endif


class Gaussian1DKernel {
public:
	Gaussian1DKernel(double regularization = 25., int W = 0) {
		gamma = regularization;
		W_ = W;
		N = W_;
		kernel1d = new double[W_];
		for (int i=0; i<W_; i++) {
			kernel1d[i] = std::max(EPSILON, exp(-i*i / gamma));
		}
	}
	Gaussian1DKernel(const Gaussian1DKernel& b) {
		gamma = b.gamma;
		W_ = b.W_;
		N = W_;
		kernel1d = new double[W_];
		memcpy(kernel1d, b.kernel1d, W_*sizeof(double));
	}
	Gaussian1DKernel& operator=(const Gaussian1DKernel& b) {
		gamma = b.gamma;
		W_ = b.W_;
		N = W_;
		kernel1d = new double[W_];
		memcpy(kernel1d, b.kernel1d, W_*sizeof(double));
		return *this;
	}
	~Gaussian1DKernel() {
		delete[] kernel1d;
	}
	double operator()(int i, int j) {
		return std::max(EPSILON, exp(-(sqr(i-j))/gamma));
	}
	void convolve(const double* u, double* result, int nvectors = 1) const {

		double* tmp = new double[W_]; // allocating here ; otherwise not thread-safe

		for (int i=0; i<nvectors; i++) {
			for (int j=0; j<W_; j++) {
				double conv = 0;
				for (int k=0; k<W_; k++) {
					conv+=kernel1d[abs(j-k)]*u[i*N+k];
				}
				tmp[j] = conv;
			}

			memcpy(result+i*N, tmp, W_*sizeof(double));
		}
		delete[] tmp;
	}
	void convolveAdjoint(const double* u, double* result, int nvectors = 1) const {
		convolve(u, result, nvectors);
	}

	double gamma;
	size_t W_, N;
	double* kernel1d;
};

class GaussianSSE1DKernel {
public:
	GaussianSSE1DKernel(double regularization = 25., int W = 0) {
		gamma = regularization;
		W_ = W;
		N = W_;
		C_ = W;
		kernel1d = (double*)malloc_simd(W*2*sizeof(double), ALIGN);
		for (int i=0; i<W*2; i++) {
			kernel1d[i] = std::max(EPSILON, exp(-sqr(C_-i) / gamma));
		}
	}
	GaussianSSE1DKernel(const GaussianSSE1DKernel& b) {
		gamma = b.gamma;
		W_ = b.W_;
		N = W_;
		C_ = W_;
    int nbthreads;
#if !defined(HAS_OPENMP)
    nbthreads=1;
#else
    nbthreads = omp_get_max_threads();
#endif
		kernel1d = (double*)malloc_simd(W_*2*sizeof(double), ALIGN);
		memcpy(kernel1d, b.kernel1d, W_*2*sizeof(double));
	}
	GaussianSSE1DKernel& operator=(const GaussianSSE1DKernel& b) {
		gamma = b.gamma;
		W_ = b.W_;
		N = W_;
		C_ = W_;
    int nbthreads;
#if !defined(HAS_OPENMP)
    nbthreads=1;
#else
    nbthreads = omp_get_max_threads();
#endif
    kernel1d = (double*)malloc_simd(W_*2*sizeof(double), ALIGN);
		memcpy(kernel1d, b.kernel1d, W_*2*sizeof(double));
		return *this;
	}
	~GaussianSSE1DKernel() {
		free_simd(kernel1d);
	}
	double operator()(int i, int j) {
		return std::max(EPSILON, exp(-(sqr(i-j))/gamma));
	}
	void convolve(const double* u, double* result, int nvectors = 1) const {

		for (int i=0; i<nvectors; i++) {

#pragma omp parallel for
			for (int j=0; j<W_; j++) {
				result[i*N+j] = dotp_full(&kernel1d[C_-j], &u[i*N], W_);
			}

		}


	}
	void convolveAdjoint(const double* u, double* result, int nvectors = 1) const {
		convolve(u, result, nvectors);
	}

	double gamma;
	size_t W_, C_;
	size_t N;
	double* kernel1d;
};


class Gaussian2DKernel {
public:
	Gaussian2DKernel(double regularization = 25., int W = 0, int H = 0) {
		gamma = regularization;
		W_ = W;
		H_ = H;
		N = W_*H_;
		kernel1d = new double[std::max(W, H)];
		for (int i=0; i<std::max(W, H); i++) {
			kernel1d[i] = std::max(EPSILON, exp(-i*i / gamma));
		}
	}
	Gaussian2DKernel(const Gaussian2DKernel& b) {
		gamma = b.gamma;
		W_ = b.W_;
		H_ = b.H_;
		N = W_*H_;
		kernel1d = new double[std::max(W_, H_)];
		memcpy(kernel1d, b.kernel1d, std::max(W_, H_)*sizeof(double));
	}
	Gaussian2DKernel& operator=(const Gaussian2DKernel& b) {
		gamma = b.gamma;
		W_ = b.W_;
		H_ = b.H_;
		N = W_*H_;
		kernel1d = new double[std::max(W_, H_)];
		memcpy(kernel1d, b.kernel1d, std::max(W_, H_)*sizeof(double));
		return *this;
	}
	~Gaussian2DKernel() {
		delete[] kernel1d;
	}
	double operator()(int i, int j) {
		int x0 = i%W_;
		int y0 = i/W_;
		int x1 = j%W_;
		int y1 = j/W_;
		return std::max(EPSILON, exp(-(sqr(x0-x1)+sqr(y0-y1))/gamma));
	}
	void convolve(const double* u, double* result, int nvectors = 1) const {

		double* tmp = new double[W_*H_]; // allocating here ; otherwise not thread-safe

		for (int nv=0; nv<nvectors; nv++) {


#pragma omp parallel for
			for (int i=0; i<H_; i++) {
				for (int j=0; j<W_; j++) {
					double conv = 0;
					for (int k=0; k<W_; k++) {
						conv+=kernel1d[abs(j-k)]*u[nv*N + i*W_ + k];
					}
					tmp[i+j*H_] = conv;
				}
			}

#pragma omp parallel for
			for (int j=0; j<W_; j++) {
				for (int i=0; i<H_; i++) {
					double conv = 0;
					for (int k=0; k<H_; k++) {
						conv+=kernel1d[abs(i-k)]*tmp[k + j*H_];
					}
					result[nv*N + i*W_+j] = conv;
				}
			}
		}
		delete[] tmp;
	}
	void convolveAdjoint(const double* u, double* result, int nvectors = 1) const {
		convolve(u, result, nvectors);
	}

	double gamma;
	size_t W_, H_;
	size_t N;
	double* kernel1d;
};



class Gaussian3DKernel {
public:
	Gaussian3DKernel(double regularization = 25., int W = 0, int H = 0, int D = 0) {
		gamma = regularization;
		W_ = W;
		H_ = H;
		D_ = D;
		N = W_*H_*D_;
		int M = std::max(std::max(W_, H_), D_);
		kernel1d = new double[M];
		for (int i=0; i<M; i++) {
			kernel1d[i] = std::max(EPSILON, exp(-i*i / gamma));
		}
	}
	Gaussian3DKernel(const Gaussian3DKernel& b) {
		gamma = b.gamma;
		W_ = b.W_;
		H_ = b.H_;
		D_ = b.D_;
		N = W_*H_*D_;
		int M = std::max(std::max(W_, H_), D_);
		kernel1d = new double[M];
		memcpy(kernel1d, b.kernel1d, M*sizeof(double));
	}
	Gaussian3DKernel& operator=(const Gaussian3DKernel& b) {
		gamma = b.gamma;
		W_ = b.W_;
		H_ = b.H_;
		D_ = b.D_;
		N = W_*H_*D_;
		int M = std::max(std::max(W_, H_), D_);
		kernel1d = new double[M];
		memcpy(kernel1d, b.kernel1d, M*sizeof(double));
		return *this;
	}
	~Gaussian3DKernel() {
		delete[] kernel1d;
	}
	double operator()(int i, int j) {
		int z0 = i/(W_*H_);
		i-=z0*W_*H_;
		int x0 = i%W_;
		int y0 = i/W_;

		int z1 = j/(W_*H_);
		j-=z1*W_*H_;
		int x1 = j%W_;
		int y1 = j/W_;
		return std::max(EPSILON, exp(-(sqr(x0-x1)+sqr(y0-y1)+sqr(z0-z1))/gamma));
	}
	void convolve(const double* u, double* result, int nvectors = 1) const {

		double* tmp = new double[W_*H_*D_]; // allocating here ; otherwise not thread-safe

		for (int nv = 0; nv< nvectors; nv++) {

#pragma omp parallel for
			for (int d=0; d<D_; d++) {
				for (int i=0; i<H_; i++) {
					for (int j=0; j<W_; j++) {
						double conv = 0;
						for (int k=0; k<W_; k++) {
							conv+=kernel1d[abs(j-k)]*u[nv*N + d*W_*H_ + i*W_ + k];
						}
						tmp[d*W_*H_ + j*H_+i] = conv;
					}
				}


				for (int j=0; j<W_; j++) {
					for (int i=0; i<H_; i++) {
						double conv = 0;
						for (int k=0; k<H_; k++) {
							conv+=kernel1d[abs(i-k)]*tmp[d*W_*H_ + j*H_ + k];
						}
						result[nv*N + (i*W_+j)*D_ + d] = conv;
					}
				}
			}
#pragma omp parallel for
			for (int i=0; i<H_; i++) {
				for (int j=0; j<W_; j++) {
					for (int d=0; d<D_; d++) {
						double conv = 0;
						for (int k=0; k<D_; k++) {
							conv+=kernel1d[abs(d-k)]*result[nv*N + k + (i*W_ + j)*D_];
						}
						tmp[d*W_*H_ + i*W_+j] = conv;
					}
				}
			}
			memcpy(result+nv*N, tmp, W_*H_*D_*sizeof(double));
		}

		delete[] tmp;
	}
	void convolveAdjoint(const double* u, double* result, int nvectors = 1) const {
		convolve(u, result, nvectors);
	}

	double gamma;
	size_t W_, H_, D_, N;
	double* kernel1d;
};




class GaussianSSE2DKernel {
public:
	GaussianSSE2DKernel(double regularization = 25., int W = 0, int H = 0) {
		gamma = regularization;
		W_ = W;
		H_ = H;
		N = W_*H_;
		C_ = std::max(W, H);
		kernel1d = (double*)malloc_simd(std::max(W, H)*2*sizeof(double), ALIGN);
		for (int i=0; i<std::max(W, H)*2; i++) {
			kernel1d[i] = std::max(EPSILON, exp(-sqr((int)C_-i) / gamma));
		}
    int nbthreads;
#if !defined(HAS_OPENMP)
    nbthreads=1;
#else
    nbthreads = omp_get_max_threads();
#endif
    tmpmem = (double*)malloc_simd(W_*H_*nbthreads*sizeof(double), ALIGN);
	}
	GaussianSSE2DKernel(const GaussianSSE2DKernel& b) {
		gamma = b.gamma;
		W_ = b.W_;
		H_ = b.H_;
		N = W_*H_;
		C_ = std::max(W_, H_);
    int nbthreads;
#if !defined(HAS_OPENMP)
    nbthreads=1;
#else
    nbthreads = omp_get_max_threads();
#endif
    kernel1d = (double*)malloc_simd(std::max(W_, H_)*2*sizeof(double), ALIGN);
		memcpy(kernel1d, b.kernel1d, std::max(W_, H_)*2*sizeof(double));
		tmpmem = (double*)malloc_simd(W_*H_*nbthreads*sizeof(double), ALIGN);
	}
	GaussianSSE2DKernel& operator=(const GaussianSSE2DKernel& b) {
		gamma = b.gamma;
		W_ = b.W_;
		H_ = b.H_;
		N = W_*H_;
		C_ = std::max(W_, H_);
		kernel1d = (double*)malloc_simd(std::max(W_, H_)*2*sizeof(double), ALIGN);
		memcpy(kernel1d, b.kernel1d, std::max(W_, H_)*2*sizeof(double));
		tmpmem = (double*)malloc_simd(W_*H_*sizeof(double), ALIGN);
		return *this;
	}
	~GaussianSSE2DKernel() {
		free_simd(kernel1d);
		free_simd(tmpmem);
	}
	double operator()(int i, int j) {
		int x0 = i%W_;
		int y0 = i/W_;
		int x1 = j%W_;
		int y1 = j/W_;
		return std::max(EPSILON, exp(-(sqr(x0-x1)+sqr(y0-y1))/gamma));
	}
	void convolve(const double* u, double* result, int nvectors = 1) const {

		double* tmp = tmpmem;

		for (int nv = 0; nv<nvectors; nv++) {
#pragma omp parallel for
			for (int i=0; i<H_; i++) {
				for (int j=0; j<W_; j++) {
					tmp[i+j*H_] = dotp_full(&kernel1d[C_-j], &u[nv*N + i*W_], W_);
				}
			}

#pragma omp parallel for
			for (int j=0; j<W_; j++) {
				for (int i=0; i<H_; i++) {
					result[nv*N + i*W_+j] = dotp_full(&kernel1d[C_-i], &tmp[j*H_], H_);
				}
			}
		}
	}
	void convolveAdjoint(const double* u, double* result, int nvectors = 1) const {
		convolve(u, result, nvectors);
	}

	double gamma;
	size_t W_, H_, C_;
	size_t N;
	double* kernel1d;
	double* tmpmem;
};


class LogGaussian2DKernel {
public:
	LogGaussian2DKernel(double regularization = 25., int W = 0, int H = 0) {
		gamma = regularization;
		W_ = W;
		H_ = H;
		N = W_*H_;
		kernel1d = new double[std::max(W, H)];
		for (int i=0; i<std::max(W, H); i++) {
			kernel1d[i] = std::max(EPSILON, exp(-i*i / gamma));
		}
	}
	LogGaussian2DKernel(const LogGaussian2DKernel& b) {
		gamma = b.gamma;
		W_ = b.W_;
		H_ = b.H_;
		N = W_*H_;
		kernel1d = new double[std::max(W_, H_)];
		memcpy(kernel1d, b.kernel1d, std::max(W_, H_)*sizeof(double));
	}
	LogGaussian2DKernel& operator=(const LogGaussian2DKernel& b) {
		gamma = b.gamma;
		W_ = b.W_;
		H_ = b.H_;
		N = W_*H_;
		kernel1d = new double[std::max(W_, H_)];
		memcpy(kernel1d, b.kernel1d, std::max(W_, H_)*sizeof(double));
		return *this;
	}
	~LogGaussian2DKernel() {
		delete[] kernel1d;
	}
	double operator()(int i, int j) {
		int x0 = i%W_;
		int y0 = i/W_;
		int x1 = j%W_;
		int y1 = j/W_;
		return std::max(EPSILON, exp(-(sqr(x0-x1)+sqr(y0-y1))/gamma));
	}
	void log_convolve(const double* log_u, double* result, int nvectors = 1) const {

		double* tmp = new double[W_*H_]; // allocating here ; otherwise not thread-safe

		for (int nv=0; nv<nvectors; nv++) {

#pragma omp parallel for
			for (int i=0; i<H_; i++) {
				for (int j=0; j<W_; j++) {
					double v_max = -1E99;
					for (int k=0; k<W_; k++) {
						v_max = std::max(v_max, -sqr(j-k)/gamma + log_u[nv*N + i*W_ + k]);
					}

					double conv = 0;
					for (int k=0; k<W_; k++) {
						conv+=exp(-sqr(j-k)/gamma + log_u[nv*N + i*W_ + k] - v_max);
					}
					tmp[i+j*H_] = log(conv) + v_max;
				}
			}

#pragma omp parallel for
			for (int j=0; j<W_; j++) {
				for (int i=0; i<H_; i++) {
					double v_max = -1E99;
					for (int k=0; k<H_; k++) {
						v_max = std::max(v_max, -sqr(i-k)/gamma + tmp[k + j*H_]);
					}

					double conv = 0;
					for (int k=0; k<H_; k++) {
						conv+=exp(-sqr(i-k)/gamma + tmp[k + j*H_] - v_max);
					}
					result[nv*N + i*W_+j] = log(conv) + v_max;
				}
			}
		}
		delete[] tmp;
	}
	void log_convolveAdjoint(const double* u, double* result, int nvectors = 1) const {
		log_convolve(u, result, nvectors);
	}

	void convolve(const double* u, double* result, int nvectors = 1) const {

		double* tmp = new double[W_*H_]; // allocating here ; otherwise not thread-safe

		for (int nv=0; nv<nvectors; nv++) {

#pragma omp parallel for
			for (int i=0; i<H_; i++) {
				for (int j=0; j<W_; j++) {
					double conv = 0;
					for (int k=0; k<W_; k++) {
						conv+=kernel1d[abs(j-k)]*u[nv*N + i*W_ + k];
					}
					tmp[i+j*H_] = conv;
				}
			}

#pragma omp parallel for
			for (int j=0; j<W_; j++) {
				for (int i=0; i<H_; i++) {
					double conv = 0;
					for (int k=0; k<H_; k++) {
						conv+=kernel1d[abs(i-k)]*tmp[k + j*H_];
					}
					result[nv*N + i*W_+j] = conv;
				}
			}
		}
		delete[] tmp;
	}
	void convolveAdjoint(const double* u, double* result, int nvectors = 1) const {
		convolve(u, result, nvectors);
	}

	double gamma;
	size_t W_, H_;
	size_t N;
	double* kernel1d;
};


class LogSignArrayGaussian2DKernel {
public:
	LogSignArrayGaussian2DKernel(double regularization = 25., int W = 0, int H = 0) {
		gamma = regularization;
		W_ = W;
		H_ = H;
		N = W_*H_;
		kernel1d = new double[std::max(W, H)];
		for (int i=0; i<std::max(W, H); i++) {
			kernel1d[i] = std::max(EPSILON, exp(-i*i / gamma));
		}
	}
	LogSignArrayGaussian2DKernel(const LogSignArrayGaussian2DKernel& b) {
		gamma = b.gamma;
		W_ = b.W_;
		H_ = b.H_;
		N = W_*H_;
		kernel1d = new double[std::max(W_, H_)];
		memcpy(kernel1d, b.kernel1d, std::max(W_, H_)*sizeof(double));
	}
	LogSignArrayGaussian2DKernel& operator=(const LogSignArrayGaussian2DKernel& b) {
		gamma = b.gamma;
		W_ = b.W_;
		H_ = b.H_;
		N = W_*H_;
		kernel1d = new double[std::max(W_, H_)];
		memcpy(kernel1d, b.kernel1d, std::max(W_, H_)*sizeof(double));
		return *this;
	}
	~LogSignArrayGaussian2DKernel() {
		delete[] kernel1d;
	}
	double operator()(int i, int j) {
		int x0 = i%W_;
		int y0 = i/W_;
		int x1 = j%W_;
		int y1 = j/W_;
		return std::max(EPSILON, exp(-(sqr(x0-x1)+sqr(y0-y1))/gamma));
	}
	void log_convolve(const double* log_u, double* result, int nvectors = 1) const {

		double* tmp = new double[W_*H_]; // allocating here ; otherwise not thread-safe

		for (int nv=0; nv<nvectors; nv++) {

#pragma omp parallel for
			for (int i=0; i<H_; i++) {
				for (int j=0; j<W_; j++) {
					double v_max = -1E99;
					for (int k=0; k<W_; k++) {
						v_max = std::max(v_max, -sqr(j-k)/gamma + log_u[nv*N + i*W_ + k]);
					}

					double conv = 0;
					for (int k=0; k<W_; k++) {
						conv+=exp(-sqr(j-k)/gamma + log_u[nv*N + i*W_ + k] - v_max);
					}
					tmp[i+j*H_] = log(conv) + v_max;
				}
			}

#pragma omp parallel for
			for (int j=0; j<W_; j++) {
				for (int i=0; i<H_; i++) {
					double v_max = -1E99;
					for (int k=0; k<H_; k++) {
						v_max = std::max(v_max, -sqr(i-k)/gamma + tmp[k + j*H_]);
					}

					double conv = 0;
					for (int k=0; k<H_; k++) {
						conv+=exp(-sqr(i-k)/gamma + tmp[k + j*H_] - v_max);
					}
					result[nv*N + i*W_+j] = log(conv) + v_max;
				}
			}
		}
		delete[] tmp;
	}
	void log_convolveAdjoint(const double* u, double* result, int nvectors = 1) const {
		log_convolve(u, result, nvectors);
	}

	void log_convolve_signArray(const double* log_u, unsigned char * signArray, double* result, int nvectors = 1) const {

		double* tmp = new double[W_*H_]; // allocating here ; otherwise not thread-safe

		unsigned char * tmpSignArray = new unsigned char[(W_*H_+7)/8];

		for (int nv=0; nv<nvectors; nv++) {

#pragma omp parallel for
			for (int i=0; i<H_; i++) {
				for (int j=0; j<W_; j++) {
					double v_max = -1E99;
					for (int k=0; k<W_; k++) {
						v_max = std::max(v_max, -sqr(j-k)/gamma + log_u[nv*N + i*W_ + k]);
					}

					double conv = 0;
					for (int k=0; k<W_; k++) {
						conv += myAbsExp(-sqr(j-k)/gamma + log_u[nv*N + i*W_ + k] - v_max, signArray,nv*N + i*W_ + k);
					}
					tmp[i+j*H_] = myAbsLog(conv, tmpSignArray, i+j*H_) + v_max;
				}
			}

#pragma omp parallel for
			for (int j=0; j<W_; j++) {
				for (int i=0; i<H_; i++) {
					double v_max = -1E99;
					for (int k=0; k<H_; k++) {
						v_max = std::max(v_max, -sqr(i-k)/gamma + tmp[k + j*H_]);
					}

					double conv = 0;
					for (int k=0; k<H_; k++) {
						conv += myAbsExp(-sqr(i-k)/gamma + tmp[k + j*H_] - v_max, tmpSignArray, k + j*H_);
					}
					result[nv*N + i*W_+j] = myAbsLog(conv, signArray, nv*N + i*W_+j) + v_max;
				}
			}
		}
		delete[] tmp;
		delete[] tmpSignArray;
	}
	void log_convolve_signArrayAdjoint(const double* u, unsigned char * sign_array, double* result, int nvectors = 1) const {
		log_convolve_signArray(u, sign_array, result, nvectors);
	}

	void convolve(const double* u, double* result, int nvectors = 1) const {

		double* tmp = new double[W_*H_]; // allocating here ; otherwise not thread-safe

		for (int nv=0; nv<nvectors; nv++) {

#pragma omp parallel for
			for (int i=0; i<H_; i++) {
				for (int j=0; j<W_; j++) {
					double conv = 0;
					for (int k=0; k<W_; k++) {
						conv+=kernel1d[abs(j-k)]*u[nv*N + i*W_ + k];
					}
					tmp[i+j*H_] = conv;
				}
			}

#pragma omp parallel for
			for (int j=0; j<W_; j++) {
				for (int i=0; i<H_; i++) {
					double conv = 0;
					for (int k=0; k<H_; k++) {
						conv+=kernel1d[abs(i-k)]*tmp[k + j*H_];
					}
					result[nv*N + i*W_+j] = conv;
				}
			}
		}
		delete[] tmp;
	}
	void convolveAdjoint(const double* u, double* result, int nvectors = 1) const {
		convolve(u, result, nvectors);
	}

	double gamma;
	size_t W_, H_;
	size_t N;
	double* kernel1d;
};


#define FP_TYPE float
#define DATA_TYPE std::complex<FP_TYPE>

class LogComplexGaussian2DKernel {
public:
	LogComplexGaussian2DKernel(FP_TYPE regularization = 25., int W = 0, int H = 0) {
		gamma = regularization;
		W_ = W;
		H_ = H;
		N = W_*H_;
		kernel1d = new FP_TYPE[std::max(W, H)];
		for (int i=0; i<std::max(W, H); i++) {
//			kernel1d[i] = std::max(EPSILON, exp(-i*i / gamma));
		}
	}
	LogComplexGaussian2DKernel(const LogComplexGaussian2DKernel& b) {
		gamma = b.gamma;
		W_ = b.W_;
		H_ = b.H_;
		N = W_*H_;
		kernel1d = new FP_TYPE[std::max(W_, H_)];
		memcpy(kernel1d, b.kernel1d, std::max(W_, H_)*sizeof(FP_TYPE));
	}
	LogComplexGaussian2DKernel& operator=(const LogComplexGaussian2DKernel& b) {
		gamma = b.gamma;
		W_ = b.W_;
		H_ = b.H_;
		N = W_*H_;
		kernel1d = new FP_TYPE[std::max(W_, H_)];
		memcpy(kernel1d, b.kernel1d, std::max(W_, H_)*sizeof(FP_TYPE));
		return *this;
	}
	~LogComplexGaussian2DKernel() {
		delete[] kernel1d;
	}
	DATA_TYPE operator()(int i, int j) {
		/*int x0 = i%W_;
		int y0 = i/W_;
		int x1 = j%W_;
		int y1 = j/W_;
		return std::max(EPSILON, exp(-(sqr(x0-x1)+sqr(y0-y1))/gamma));*/

		return 0;
	}
	void log_convolve(const DATA_TYPE* log_u, DATA_TYPE* result, int nvectors = 1) const {

		DATA_TYPE* tmp = new DATA_TYPE[W_*H_]; // allocating here ; otherwise not thread-safe

		for (int nv=0; nv<nvectors; nv++) {

#pragma omp parallel for
			for (int i=0; i<H_; i++) {
				for (int j=0; j<W_; j++) {
					DATA_TYPE v_max = 1E-30;
//					DATA_TYPE v_mean = 0;
					for (int k=0; k<W_; k++) {
						v_max = std::max(std::real(v_max), std::real(-sqr(j-k)/gamma + log_u[nv*N + i*W_ + k]));
//						v_mean += -sqr(j-k)/gamma + log_u[nv*N + i*W_ + k];
					}
//					v_mean/=W_;

					DATA_TYPE conv = 0;
					for (int k=0; k<W_; k++) {
						conv+=std::exp(-sqr(j-k)/gamma + log_u[nv*N + i*W_ + k] - v_max);
//						conv+=exp(-sqr(j-k)/gamma + log_u[nv*N + i*W_ + k] - v_mean);
					}
					tmp[i+j*H_] = std::log(conv) + v_max;
//					tmp[i+j*H_] = log(conv) + v_mean;
				}
			}

#pragma omp parallel for
			for (int j=0; j<W_; j++) {
				for (int i=0; i<H_; i++) {
					DATA_TYPE v_max = 1E-30;
//					DATA_TYPE v_mean = 0;
					for (int k=0; k<H_; k++) {
						v_max = std::max(std::real(v_max),std::real(-sqr(i-k)/gamma + tmp[k + j*H_]));
//						v_mean += -sqr(i-k)/gamma + tmp[k + j*H_];
					}
//					v_mean/=H_;

					DATA_TYPE conv = 0;
					for (int k=0; k<H_; k++) {
						conv+=std::exp(-sqr(i-k)/gamma + tmp[k + j*H_] - v_max);
//						conv+=exp(-sqr(i-k)/gamma + tmp[k + j*H_] - v_mean);
					}
					result[nv*N + i*W_+j] = std::log(conv) + v_max;
//					result[nv*N + i*W_+j] = log(conv) + v_mean;
				}
			}
		}
		delete[] tmp;
	}
	void log_convolveAdjoint(const DATA_TYPE* u, DATA_TYPE* result, int nvectors = 1) const {
		log_convolve(u, result, nvectors);
	}

	void log_convolve(const double* log_u, double* result, int nvectors = 1) const {

		double* tmp = new double[W_*H_]; // allocating here ; otherwise not thread-safe

		for (int nv=0; nv<nvectors; nv++) {

#pragma omp parallel for
			for (int i=0; i<H_; i++) {
				for (int j=0; j<W_; j++) {
					double v_max = -1E99;
//					double v_mean = 0;
					for (int k=0; k<W_; k++) {
						v_max = std::max(v_max, -sqr(j-k)/gamma + log_u[nv*N + i*W_ + k]);
//						v_mean += -sqr(j-k)/gamma + log_u[nv*N + i*W_ + k];
					}
//					v_mean/=W_;

					double conv = 0;
					for (int k=0; k<W_; k++) {
						conv+=exp(-sqr(j-k)/gamma + log_u[nv*N + i*W_ + k] - v_max);
//						conv+=exp(-sqr(j-k)/gamma + log_u[nv*N + i*W_ + k] - v_mean);
					}
					tmp[i+j*H_] = log(conv) + v_max;
//					tmp[i+j*H_] = log(conv) + v_mean;
				}
			}

#pragma omp parallel for
			for (int j=0; j<W_; j++) {
				for (int i=0; i<H_; i++) {
					double v_max = -1E99;
//					double v_mean = 0;
					for (int k=0; k<H_; k++) {
						v_max = std::max(v_max, -sqr(i-k)/gamma + tmp[k + j*H_]);
//						v_mean += -sqr(i-k)/gamma + tmp[k + j*H_];
					}
//					v_mean/=H_;

					double conv = 0;
					for (int k=0; k<H_; k++) {
						conv+=exp(-sqr(i-k)/gamma + tmp[k + j*H_] - v_max);
//						conv+=exp(-sqr(i-k)/gamma + tmp[k + j*H_] - v_mean);
					}
					result[nv*N + i*W_+j] = log(conv) + v_max;
//					result[nv*N + i*W_+j] = log(conv) + v_mean;
				}
			}
		}
		delete[] tmp;
	}
	void log_convolveAdjoint(const double* u, double* result, int nvectors = 1) const {
		log_convolve(u, result, nvectors);
	}

	// Keep the normal convolution function for compatibility
	void convolve(const double* u, double* result, int nvectors = 1) const {

		double* tmp = new double[W_*H_]; // allocating here ; otherwise not thread-safe

		for (int nv=0; nv<nvectors; nv++) {


#pragma omp parallel for
			for (int i=0; i<H_; i++) {
				for (int j=0; j<W_; j++) {
					double conv = 0;
					for (int k=0; k<W_; k++) {
						conv+=kernel1d[abs(j-k)]*u[nv*N + i*W_ + k];
					}
					tmp[i+j*H_] = conv;
				}
			}

#pragma omp parallel for
			for (int j=0; j<W_; j++) {
				for (int i=0; i<H_; i++) {
					double conv = 0;
					for (int k=0; k<H_; k++) {
						conv+=kernel1d[abs(i-k)]*tmp[k + j*H_];
					}
					result[nv*N + i*W_+j] = conv;
				}
			}
		}
		delete[] tmp;
	}
	void convolveAdjoint(const double* u, double* result, int nvectors = 1) const {
		convolve(u, result, nvectors);
	}

	FP_TYPE gamma;
	size_t W_, H_;
	size_t N;
	FP_TYPE* kernel1d;
};


#ifdef HAS_HALIDE


static inline Func gaussian_convol(ImageParam &in, Param<double> &gamma) {
	Var x("x"), y("y");

	Func k;
	k(x) = cast<double>(Halide::max((float)EPSILON, exp(-(x*x)/gamma)));

	k.compute_root().memoize();


	RDom rx(0, in.width());
	Func blur_x1("blur_x1");
	blur_x1(x, y) = cast<double>(0);
	blur_x1(x, y) += in(rx, y) * k(x - rx);

	Func transpose1("transpose1");
	transpose1(x, y) = blur_x1(y, x);

	RDom ry(0, in.height());
	Func blur_x2("blur_x2");
	blur_x2(x, y) = cast<double>(0);
	blur_x2(x, y) += transpose1(ry, y) * k(x - ry);

	Func transpose2("transpose2");;
	transpose2(x, y) = blur_x2(y, x);

	Var xi, yi, xo, yo, tile;

	transpose1.compute_root().tile(x, y, xi, yi, 4,
		8).vectorize(xi).reorder(y, x).parallel(x);
	blur_x1.compute_at(transpose1, y).vectorize(x);
	blur_x1.update().vectorize(x).reorder(y, rx).unroll(y);

	transpose2.compute_root().tile(x, y, xi, yi, 4,
		8).vectorize(xi).reorder(y, x).parallel(x);
	blur_x2.compute_at(transpose2, y).vectorize(x);
	blur_x2.update().vectorize(x).reorder(y, ry).unroll(y);

	return transpose2;
}
static inline Func gaussian_convol_multiple(ImageParam &in, Param<double> &gamma) {
	Var x("x"), y("y"), z("z"), t("t"), t2("t2");

	Func k;
	k(x) = cast<double>(Halide::max((float)EPSILON, exp(-(x*x)/gamma)));

	k.compute_root().memoize();

	const int vec_size = 4;

	Var ti[3], tj[3];
	Func resultX("resultX"), resultY("resultY");

	const Expr W = in.width();
	const Expr H = in.height();
	const Expr K = in.channels();
	const Expr sum_size_vec = W / vec_size;
	const Expr sum_size_vecY = H / vec_size;

	Var w("w");
	Func prod("prod"), prodY("prodY"), dot_vecs("dot_vecs"), dot_vecsY("dot_vecsY"), dot_vecs_transpose("dot_vecs_transpose"), dot_vecs_transposeY("dot_vecs_transposeY"), sum_lanes("sum_lanes"),
		sum_lanesY("sum_lanesY"), sum_tail("sum_tail"), sum_tailY("sum_tailY"), AB("AB"), ABY("ABY");
	RDom rv(0, sum_size_vec), lanes(0, vec_size);
	RDom tail(sum_size_vec * vec_size, W - sum_size_vec * vec_size);

	prod(w, x, y, z) = in(w, y, z) * k(x - w);
	dot_vecs(w, x, y, z) = cast<double>(0);
	dot_vecs(w, x, y, z) += prod(rv * vec_size + w, x, y, z);
	dot_vecs_transpose(x, y, z, w) = dot_vecs(w, x, y, z);

	sum_lanes(x, y, z) += dot_vecs_transpose(x, y, z, lanes);
	sum_tail(x, y, z) = cast<double>(0);
	sum_tail(x, y, z) += prod(tail, x, y, z);
	AB(x, y, z) = sum_lanes(x, y, z) + sum_tail(x, y, z);
	resultX(x, y, z) = AB(x, y, z);

	Func transpose2("transpose2");;
	Func transpose1("transpose1");

	RDom rvY(0, sum_size_vecY), lanesY(0, vec_size);
	RDom tailY(sum_size_vecY * vec_size, H - sum_size_vecY * vec_size);

	/*
	transpose1(x, y, z) = resultX(y, x, z);

	prodY(w, x, y, z) = transpose1(w, y, z) * k(x - w);
	dot_vecsY(w, x, y, z) += prodY(rvY * vec_size + w, x, y, z);
	dot_vecs_transposeY(x, y, z, w) = dot_vecsY(w, x, y, z);

	sum_lanesY(x, y, z) += dot_vecs_transposeY(x, y, z, lanes);
	sum_tailY(x, y, z) = cast<double>(0);
	sum_tailY(x, y, z) += prod(tailY, x, y, z);
	ABY(x, y, z) = sum_lanes(x, y, z) + sum_tail(x, y, z);
	resultY(x,y,z) = ABY(x,y,z);



	transpose2(x, y, z) = resultY(y, x, z);*/


	Var xi, yi, zi, tii, xo, yo, tile;
	/*transpose1.compute_root().tile(x, y, xi, yi, 4,
	8).vectorize(xi).reorder(y, x).parallel(z);
	transpose2.compute_root().tile(x, y, xi, yi, 4,
	8).vectorize(xi).reorder(y, x).parallel(z);*/



	/*resultX.parallel(y).reorder(z, y).reorder(x, z)
	.specialize(W == (W / 8) * 8)
	.specialize(W >= 4 && K >= 2)
	.tile(x, z, xi, zi, 4, 2).vectorize(zi).unroll(zi)
	.specialize(W >= 8 && K >= 8)
	.tile(x, z, ti[0], tj[0], x, z, 2, 4)
	.specialize(W >= 16 && K >= 16)
	.tile(ti[0], tj[0], ti[1], tj[1], 2, 2)
	.specialize(W >= 32 && K >= 32)
	.tile(ti[0], tj[0], ti[2], tj[2], 2, 2)
	.specialize(W >= 64 && K >= 64)
	;*/



	// The general case with a tail (sum_size is not a multiple of
	// vec_size). The same z-order traversal of blocks of the
	// output.
	resultX.fuse(y, z, t)
		//.specialize(W >= 4 && K >= 2)
		.tile(x, t, xi, tii, 4, 2).vectorize(xi).unroll(tii)
		.specialize(W >= 8 && K*H >= 8)
		.tile(x, t, ti[0], tj[0], x, t, 2, 4)
		.specialize(W >= 16 && K*H >= 16)
		.tile(ti[0], tj[0], ti[1], tj[1], 2, 2)
		.specialize(W >= 32 && K*H >= 32)
		.tile(ti[0], tj[0], ti[2], tj[2], 2, 2)
		.specialize(W >= 64 && K*H >= 64).fuse(tj[0], ti[0], t2).parallel(t2);
	;

	dot_vecs.fuse(y, z, t)
		.compute_at(resultX, x).unroll(x).unroll(t, 8)
		.update().fuse(y, z, t).reorder(x, t, rv).unroll(x).unroll(t, 8);
	dot_vecs_transpose.fuse(y, z, t)
		.compute_at(resultX, x).unroll(x).unroll(t, 8);
	sum_lanes
		.compute_at(resultX, x).update().unroll(lanes);
	sum_tail
		.compute_at(resultX, x)
		.update().fuse(y, z, t).reorder(x, t, tail).unroll(x).unroll(t, 8);


	dot_vecs.vectorize(w).update().vectorize(w);
	dot_vecs_transpose.vectorize(w);


	// The following stages are only vectorizable when we're
	// computing multiple dot products unrolled.
	Expr can_vectorize = W >= 4 && K >= 2;
	sum_tail.specialize(can_vectorize).fuse(x, z, t2).vectorize(t2, 4);
	sum_lanes.specialize(can_vectorize).fuse(x, z, t2).vectorize(t2, 4);
	sum_lanes.update().specialize(can_vectorize).fuse(x, z, t2).vectorize(t2, 4);



	/*// The general case with a tail (sum_size is not a multiple of
	// vec_size). The same z-order traversal of blocks of the
	// output.
	resultY
	.specialize(H >= 4 && K >= 2)
	.tile(x, z, xi, zi, 4, 2).vectorize(xi).unroll(zi)
	.specialize(H >= 8 && K >= 8)
	.tile(x, z, ti[0], tj[0], x, z, 2, 4)
	.specialize(H >= 16 && K >= 16)
	.tile(ti[0], tj[0], ti[1], tj[1], 2, 2)
	.specialize(H >= 32 && K >= 32)
	.tile(ti[0], tj[0], ti[2], tj[2], 2, 2)
	.specialize(H >= 64 && K >= 64)
	.fuse(tj[0], ti[0], t).parallel(t);

	dot_vecsY
	.compute_at(resultY, x).unroll(x).unroll(z)
	.update().reorder(x, z, rvY).unroll(x).unroll(z);
	dot_vecs_transposeY
	.compute_at(resultY, x).unroll(x).unroll(z);
	sum_lanesY
	.compute_at(resultY, x).update().unroll(lanes);
	sum_tailY
	.compute_at(resultY, x)
	.update().reorder(x, z, tailY).unroll(x).unroll(z);


	dot_vecsY.vectorize(w).update().vectorize(w);
	dot_vecs_transposeY.vectorize(w);


	// The following stages are only vectorizable when we're
	// computing multiple dot products unrolled.
	Expr can_vectorizeY = H >= 4 && K >= 2;
	sum_tailY.specialize(can_vectorizeY).fuse(x, z, t).vectorize(t);
	sum_lanesY.specialize(can_vectorizeY).fuse(x, z, t).vectorize(t);
	sum_lanesY.update().specialize(can_vectorizeY).fuse(x, z, t).vectorize(t);*/


	in.set_min(0, 0).set_min(1, 0);

	return resultX;// transpose2;
}

static inline Func gaussian_convol_3d(ImageParam &in, Param<double> &gamma) {
	Var x("x"), y("y"), z("z");

	Func k;
	k(x) = cast<double>(Halide::max((float)EPSILON, exp(-(x*x)/gamma)));

	k.compute_root().memoize();

	RDom rx(0, in.width());
	Func blur_x1("blur_x1");
	blur_x1(x, y, z) = cast<double>(0);
	blur_x1(x, y, z) += in(rx, y, z) * k(x - rx);

	Func transpose1("transpose1");
	transpose1(x, y, z) = blur_x1(y, x, z);

	RDom ry(0, in.height());
	Func blur_x2("blur_x2");
	blur_x2(x, y, z) = cast<double>(0);
	blur_x2(x, y, z) += transpose1(ry, y, z) * k(x - ry);

	Func transpose2("transpose2");
	transpose2(x, y, z) = blur_x2(y, z, x);

	RDom rz(0, in.channels());
	Func blur_x3("blur_x3");
	blur_x3(x, y, z) = cast<double>(0);
	blur_x3(x, y, z) += transpose2(rz, y, z) * k(x - rz);

	Func transpose3("transpose3");
	transpose3(x, y, z) = blur_x3(z, y, x);

	Var xi, yi, zi, xo, yo, tile;

	transpose1.compute_root().tile(x, y, xi, yi, 4,
		8).vectorize(xi).reorder(y, x).parallel(z);
	blur_x1.compute_at(transpose1, y).vectorize(x);
	blur_x1.update().vectorize(x).reorder(y, rx).unroll(y);

	transpose2.compute_root().tile(x, y, xi, yi, 4,
		8).vectorize(xi).reorder(y, x).parallel(z);
	blur_x2.compute_at(transpose2, y).vectorize(x);
	blur_x2.update().vectorize(x).reorder(y, ry).unroll(y);

	transpose3.compute_root().tile(x, z, xi, zi, 4,
		8).vectorize(xi).reorder(z, y).parallel(y);
	blur_x3.compute_at(transpose3, z).vectorize(x);
	blur_x3.update().vectorize(x).reorder(z, rz).unroll(z);

	//transpose3.compile_to_lowered_stmt("testout.html", {}, HTML);
	return transpose3;
}

class GaussianHalide2DKernel {
public:
	GaussianHalide2DKernel(double regularization = 25., int W = 0, int H = 0) {
		gamma = regularization;
		W_ = W;
		H_ = H;
		N = W_*H_;
		in = new ImageParam(type_of<double>(), 2);
		in3d = new ImageParam(type_of<double>(), 3);
		regul = new Param<double>("regul");
		gauss = gaussian_convol(*in, *regul);
//		gauss_multiple = gaussian_convol_multiple(*in3d, *regul);
		Target target = get_host_target();

		gauss.compile_jit(target);
//		gauss_multiple.compile_jit(target);

	}
	GaussianHalide2DKernel(const GaussianHalide2DKernel& b) {
		regul = b.regul;
		in = b.in; //beware, copies pointers
		in3d = b.in3d; //beware, copies pointers
		gamma = b.gamma;
		W_ = b.W_;
		H_ = b.H_;
		N = W_*H_;
		gauss = b.gauss;
//		gauss_multiple = b.gauss_multiple;
	}
	GaussianHalide2DKernel& operator=(const GaussianHalide2DKernel& b) {
		regul = b.regul;
		in = b.in; //beware, copies pointers
		gamma = b.gamma;
		in3d = b.in3d; //beware, copies pointers
		W_ = b.W_;
		H_ = b.H_;
		N = W_*H_;
		gauss = b.gauss;
//		gauss_multiple = b.gauss_multiple;
		return *this;
	}
	~GaussianHalide2DKernel() {

	}
	double operator()(int i, int j) {
		int x0 = i%W_;
		int y0 = i/W_;
		int x1 = j%W_;
		int y1 = j/W_;
		return std::max(EPSILON, exp(-(sqr(x0-x1)+sqr(y0-y1))/gamma));
	}
	void convolve(const double* u, double* result, int nvectors = 1) const {
//		if (nvectors<100000) {
			for (int i=0; i<nvectors; i++) {
				Bufferd resbuff((result + i*N), W_, H_);
				Bufferd ubuff(const_cast<double*>(u + i*N), W_, H_);
				in->set(ubuff);
				regul->set(gamma);
				gauss.realize(resbuff);
			}
//		}
//		else {
//			Bufferd resbuff(result, W_, H_, nvectors);
//			Bufferd ubuff(const_cast<double*>(u), W_, H_, nvectors);
//			in3d->set(ubuff);
//			regul->set(gamma);
//			gauss_multiple.realize(resbuff);
//		}
	}
	void convolveAdjoint(const double* u, double* result, int nvectors = 1) const {
		convolve(u, result, nvectors);
	}

	double gamma;
	size_t W_, H_;
	size_t N;
	ImageParam *in;
	ImageParam *in3d;
	Param<double> *regul;
	mutable Func gauss;
//	mutable Func gauss_multiple;
};



class GaussianHalide3DKernel {
public:
	GaussianHalide3DKernel(double regularization = 25., int W = 0, int H = 0, int D = 0) {
		gamma = regularization;
		W_ = W;
		H_ = H;
		D_ = D;
		N = W_*H_*D_;
		in = new ImageParam(type_of<double>(), 3);
		regul = new Param<double>("regul");
		gauss = gaussian_convol_3d(*in, *regul);
		Target target = get_host_target();
		gauss.compile_jit(target);
	}
	GaussianHalide3DKernel(const GaussianHalide3DKernel& b) {
		regul = b.regul;
		in = b.in; //beware, copies pointers
		gamma = b.gamma; //beware, copies pointers
		W_ = b.W_;
		H_ = b.H_;
		D_ = b.D_;
		N = W_*H_*D_;
		gauss = b.gauss;
	}
	GaussianHalide3DKernel& operator=(const GaussianHalide3DKernel& b) {
		regul = b.regul;
		in = b.in; //beware, copies pointers
		gamma = b.gamma; //beware, copies pointers
		W_ = b.W_;
		H_ = b.H_;
		D_ = b.D_;
		N = W_*H_*D_;
		gauss = b.gauss;
		return *this;
	}
	~GaussianHalide3DKernel() {

	}
	double operator()(int i, int j) {
		int z0 = i/(W_*H_);
		i-=z0*W_*H_;
		int x0 = i%W_;
		int y0 = i/W_;

		int z1 = j/(W_*H_);
		j-=z1*W_*H_;
		int x1 = j%W_;
		int y1 = j/W_;
		return std::max(EPSILON, exp(-(sqr(x0-x1)+sqr(y0-y1)+sqr(z0-z1))/gamma));
	}
	void convolve(const double* u, double* result, int nvectors = 1) const {

		for (int i=0; i<nvectors; i++) {
			Bufferd resbuff((result + i*N), W_, H_, D_);
			Bufferd ubuff(const_cast<double*>(u + i*N), W_, H_, D_);
			in->set(ubuff);
			regul->set(gamma);
			gauss.realize(resbuff);
		}
	}
	void convolveAdjoint(const double* u, double* result, int nvectors = 1) const {
		convolve(u, result, nvectors);
	}

	double gamma;
	size_t W_, H_, D_;
	size_t N;
	ImageParam *in;
	Param<double> *regul;
	mutable Func gauss;
};

#endif


class GaussianSSE3DKernel {
public:
	GaussianSSE3DKernel(double regularization = 25., int W = 0, int H = 0, int D = 0) {
		gamma = regularization;
		W_ = W;
		H_ = H;
		D_ = D;
		N = W_*H_*D_;
		int M = std::max(std::max(W_, H_), D_)*2;
		C_ = M/2;
		kernel1d = (double*)malloc_simd(M*sizeof(double), ALIGN);
		for (int i=0; i<M; i++) {
			kernel1d[i] = std::max(EPSILON, exp(-sqr(i-(int)C_) / gamma));
		}
		tmpmem = (double*)malloc_simd(W_*H_*D_*sizeof(double), ALIGN);
	}
	GaussianSSE3DKernel(const GaussianSSE3DKernel& b) {
		gamma = b.gamma;
		W_ = b.W_;
		H_ = b.H_;
		D_ = b.D_;
		N = W_*H_*D_;
		int M = std::max(std::max(W_, H_), D_)*2;
		C_ = M/2;
		kernel1d = (double*)malloc_simd(M*sizeof(double), ALIGN);
		memcpy(kernel1d, b.kernel1d, M*sizeof(double));
		tmpmem = (double*)malloc_simd(W_*H_*D_*sizeof(double), ALIGN);
	}
	GaussianSSE3DKernel& operator=(const GaussianSSE3DKernel& b) {
		gamma = b.gamma;
		W_ = b.W_;
		H_ = b.H_;
		D_ = b.D_;
		N = W_*H_*D_;
		int M = std::max(std::max(W_, H_), D_)*2;
		C_ = M/2;
		kernel1d = (double*)malloc_simd(M*sizeof(double), ALIGN);
		memcpy(kernel1d, b.kernel1d, M*sizeof(double));
		tmpmem = (double*)malloc_simd(W_*H_*D_*sizeof(double), ALIGN);

		return *this;
	}
	~GaussianSSE3DKernel() {
		free_simd(tmpmem);
		free_simd(kernel1d);
	}
	double operator()(int i, int j) {
		int z0 = i/(W_*H_);
		i-=z0*W_*H_;
		int x0 = i%W_;
		int y0 = i/W_;

		int z1 = j/(W_*H_);
		j-=z1*W_*H_;
		int x1 = j%W_;
		int y1 = j/W_;
		return std::max(EPSILON, exp(-(sqr(x0-x1)+sqr(y0-y1)+sqr(z0-z1))/gamma));
	}
	void convolve(const double* u, double* result, int nvectors = 1) const {

		double* tmp = tmpmem;
		for (int nv=0; nv<nvectors; nv++) {
#pragma omp parallel for
			for (int d=0; d<D_; d++) {
				for (int i=0; i<H_; i++) {
					const double* ud = &u[nv*N + d*W_*H_ + i*W_];
					for (int j=0; j<W_; j++) {
						tmp[d*W_*H_ + j*H_+i] = dotp_full(&kernel1d[C_-j], ud, W_);
					}
				}


				for (int j=0; j<W_; j++) {
					const double* ud = &tmp[d*W_*H_ + j*H_];
					for (int i=0; i<H_; i++) {
						result[nv*N + (i*W_+j)*D_ + d] = dotp_full(&kernel1d[C_-i], ud, H_);
					}
				}
			}

#pragma omp parallel for
			for (int i=0; i<H_; i++) {
				for (int j=0; j<W_; j++) {
					const double* ud = &result[(i*W_ + j)*D_];
					for (int d=0; d<D_; d++) {
						tmp[d*W_*H_ + i*W_+j] = dotp_full(&kernel1d[C_-d], ud, D_);
					}
				}
			}
			memcpy(result+nv*N, tmp, W_*H_*D_*sizeof(double));
		}
	}
	void convolveAdjoint(const double* u, double* result, int nvectors = 1) const {
		convolve(u, result, nvectors);
	}

	double gamma;
	size_t W_, H_, D_, N, C_;
	double* kernel1d;
	double* tmpmem;
};



#ifdef HAS_HALIDE


static inline Func log_gaussian_convol(ImageParam &in, Param<double> &gamma) {
	Var x("x"), y("y");

	Func sq("sq");
	sq(x) = x*x;
	RDom rx(0, in.width());
	Func vmaxx("max1");
	vmaxx(x, y) = Halide::maximum(in(rx, y)  - sq(rx - x) / gamma);

	Func blur_x1("blur_x1");
	blur_x1(x, y) = cast<double>(0);
	blur_x1(x, y) += Halide::exp(in(rx, y)  - sq(rx -x)/gamma - vmaxx(x,y));

	Func transpose1("transpose1");
	transpose1(x, y) = log(blur_x1(y, x)) + vmaxx(y,x);

	RDom ry(0, in.height());
	Func vmaxy("max2");
	vmaxy(x, y) = Halide::maximum(transpose1(ry, y)  - sq(ry - x) / gamma);

	Func blur_x2("blur_x2");
	blur_x2(x, y) = cast<double>(0);
	blur_x2(x, y) += Halide::exp(transpose1(ry, y)  - sq(ry-x)/gamma - vmaxy(x,y));
	blur_x2(x, y) = log(blur_x2(x, y)) + vmaxy(x, y);


	Func transpose2("transpose2");
	transpose2(x, y) = blur_x2(y, x);

	Var xi, yi, xo, yo, tile;

	transpose1.compute_root().tile(x, y, xi, yi, 1,
		4).reorder(y, x).parallel(x);
	vmaxx.compute_at(transpose1, y).vectorize(x);
	blur_x1.compute_at(transpose1, y).vectorize(x);
	blur_x1.update().vectorize(x).reorder(y, rx).unroll(y);

	transpose2.compute_root().tile(x, y, xi, yi, 1,
		4).reorder(y, x).parallel(x);
	vmaxy.compute_at(transpose2, y).vectorize(x);
	blur_x2.compute_at(transpose2, y).vectorize(x);
	blur_x2.update().vectorize(x).reorder(y, ry).unroll(y);;

	return transpose2;
}

static inline Func log_gaussian_convol(ImageParam &in, Param<float> &gamma) {
	Var x("x"), y("y");

	Func sq("sq");
	sq(x) = x*x;
	RDom rx(0, in.width());
	Func vmaxx("max1");
	vmaxx(x, y) = Halide::maximum(in(rx, y) - sq(rx - x) / gamma);

	Func blur_x1("blur_x1");
	blur_x1(x, y) = cast<float>(0);
	blur_x1(x, y) += Halide::exp(in(rx, y) - sq(rx - x) / gamma - vmaxx(x, y));

	Func transpose1("transpose1");
	transpose1(x, y) = Halide::log(blur_x1(y, x)) + vmaxx(y, x);

	RDom ry(0, in.height());
	Func vmaxy("max2");
	vmaxy(x, y) = Halide::maximum(transpose1(ry, y) - sq(ry - x) / gamma);

	Func blur_x2("blur_x2");
	blur_x2(x, y) = cast<float>(0);
	blur_x2(x, y) += Halide::exp(transpose1(ry, y) - sq(ry - x) / gamma - vmaxy(x, y));
	blur_x2(x, y) = Halide::log(blur_x2(x, y)) + vmaxy(x, y);


	Func transpose2("transpose2");
	transpose2(x, y) = blur_x2(y, x);

	Var xi, yi, xo, yo, tile;

	transpose1.compute_root().tile(x, y, xi, yi, 1,
		8).reorder(y, x).parallel(x);
	vmaxx.compute_at(transpose1, y).vectorize(x);
	blur_x1.compute_at(transpose1, y).vectorize(x);
	blur_x1.update().vectorize(x).reorder(y, rx).unroll(y);

	transpose2.compute_root().tile(x, y, xi, yi, 1,
		8).reorder(y, x).parallel(x);
	vmaxy.compute_at(transpose2, y).vectorize(x);
	blur_x2.compute_at(transpose2, y).vectorize(x);
	blur_x2.update().vectorize(x).reorder(y, ry).unroll(y);;

	return transpose2;
}



// several issues: - cannot output two results in one Halide program ( https://github.com/halide/Halide/issues/1529 )
// - scheduling issues right now, could potentially be fixed I guess
template<typename T>
static inline Func log_gaussian_convol_sign(ImageParam &in, ImageParam &signs, Param<T> &gamma) {
	Var x("x"), y("y"), i("i");

	Func getsign("getsign");
	getsign(x,y) = ((signs(x / 8, y) >> (x % 8)) & 1)*2-1;

	Func sq("sq");
	sq(x) = x*x;
	RDom rx(0, in.width());
	Func vmaxx("max1");
	vmaxx(x, y) = Halide::maximum(in(rx, y) - sq(rx - x) / gamma);

	// for blur_x1 y
	//  for blur_x1 x
	//   float maxi
	//   for range k
	//     maxi = max(maxi, .)
	//   float blur = 0
	//   for range k
	//      blur += ...


	//   tmpsign
	//   transpose(x,y) = log(abs(blur))+max



	Func blur_x1("blur_x1");
	blur_x1(x, y) = cast<double>(0);
	blur_x1(x, y) += getsign(x,y)*exp(in(rx, y) - sq(rx - x) / gamma - vmaxx(x, y));
	Func tmpsign("tmpsign");
	tmpsign(x,y) = Halide::select(blur_x1(x*8,y)>0,1,0) + Halide::select(blur_x1(x * 8+1, y)>0, 2, 0) + Halide::select(blur_x1(x * 8 + 2, y)>0, 4, 0) + Halide::select(blur_x1(x * 8 + 3, y)>0, 8, 0)
		+ Halide::select(blur_x1(x * 8 + 4, y)>0, 16, 0) + Halide::select(blur_x1(x * 8 + 5, y)>0, 32, 0) + Halide::select(blur_x1(x * 8 + 6, y)>0, 64, 0) + Halide::select(blur_x1(x * 8 + 7, y)>0, 128, 0);

	Func getsigntmp("getsigntmp");
	getsigntmp(x,y) = ((tmpsign(x/8,y) >> (x % 8)) & 1) * 2 - 1;

	//Func myabsexptmp("myabsexptmp");
	//myabsexptmp(x, i) = getsigntmp(i + id*in.height()*in.width()) * exp(x);

	Func transpose1("transpose1");
	transpose1(x, y) = log(abs(blur_x1(y, x))) + vmaxx(y, x);

	RDom ry(0, in.height());
	Func vmaxy("max2");
	vmaxy(x, y) = Halide::maximum(transpose1(ry, y) - sq(ry - x) / gamma);


	Func blur_x2("blur_x2");
	blur_x2(x, y) = cast<double>(0);
	blur_x2(x, y) += getsigntmp(x,y)*exp(transpose1(ry, y) - sq(ry - x) / gamma - vmaxy(x, y));

	signs(x, y) = Halide::select(blur_x2(x * 8, y)>0, 1, 0) + Halide::select(blur_x2(x * 8 + 1, y)>0, 2, 0) + Halide::select(blur_x2(x * 8 + 2, y)>0, 4, 0) + Halide::select(blur_x2(x * 8 + 3, y)>0, 8, 0)
		+ Halide::select(blur_x2(x * 8 + 4, y)>0, 16, 0) + Halide::select(blur_x2(x * 8 + 5, y)>0, 32, 0) + Halide::select(blur_x2(x * 8 + 6, y)>0, 64, 0) + Halide::select(blur_x2(x * 8 + 7, y)>0, 128, 0);

	Func transpose2("transpose2");
	transpose2(x, y) = log(abs(blur_x2(y, x))) + vmaxy(y, x);

	Var xi, yi, xo, yo;

	blur_x1.compute_root();
	vmaxx.compute_at(blur_x1, x).store_root();
	tmpsign.compute_root();
	transpose1.compute_root();


	transpose2.compute_root().reorder(y, x);
	vmaxy.compute_at(transpose2, y);
	blur_x2.compute_at(transpose2, y);
	blur_x2.update().reorder(y, ry);
	//signs.compute_at(blur_x2 x);

	/*transpose1.compute_root().tile(x, y, xi, yi, 1,
		4).reorder(y, x).parallel(x);
	vmaxx.compute_at(transpose1, y).vectorize(x);
	blur_x1.compute_at(transpose1, y).vectorize(x);
	blur_x1.update().vectorize(x).reorder(y, rx).unroll(y);
	tmpsign.compute_at(blur_x1, x);*/

	/*transpose2.compute_root().tile(x, y, xi, yi, 1,
		4).reorder(y, x).parallel(x);
	vmaxy.compute_at(transpose2, y).vectorize(x);
	blur_x2.compute_at(transpose2, y).vectorize(x);
	blur_x2.update().vectorize(x).reorder(y, ry).unroll(y);;*/

	return transpose2;
}

template<typename T>
class LogSignArrayGaussianHalide2DKernel {
public:
	LogSignArrayGaussianHalide2DKernel(T regularization = 25.f, int W = 0, int H = 0) {
			gamma = regularization;
			W_ = W;
			H_ = H;
			N = W_*H_;
			in = new ImageParam(type_of<T>(), 2);
			signs = new ImageParam(type_of<unsigned char>(), 2);
			regul = new Param<T>("regul");
			id = new Param<int>("id");
			gauss = log_gaussian_convol(*in, *regul);
			//gauss_sign = log_gaussian_convol_sign(*in, *signs, *regul);  //not functional
			Target target = get_host_target();

			gauss.compile_jit(target);
			//gauss_sign.compile_jit(target); //not functional

			kernel1d = new T[std::max(W, H)];
			for (int i = 0; i<std::max(W, H); i++) {
				kernel1d[i] = std::max((T)EPSILON, (T) exp(-i*i / gamma));
			}
		}
	LogSignArrayGaussianHalide2DKernel(const LogSignArrayGaussianHalide2DKernel<T>& b) {
		regul = b.regul;
		in = b.in; //beware, copies pointers
		gamma = b.gamma;
		W_ = b.W_;
		H_ = b.H_;
		N = W_*H_;
		gauss = b.gauss;
	}


	LogSignArrayGaussianHalide2DKernel& operator=(const LogSignArrayGaussianHalide2DKernel<T>& b) {
		gamma = b.gamma;
		W_ = b.W_;
		H_ = b.H_;
		N = W_*H_;
		regul = b.regul;
		in = b.in; //beware, copies pointers
		gauss = b.gauss;
		kernel1d = new T[std::max(W_, H_)];
		memcpy(kernel1d, b.kernel1d, std::max(W_, H_) * sizeof(T));
		return *this;
	}
	~LogSignArrayGaussianHalide2DKernel() {
		//delete[] kernel1d;
	}
	T operator()(int i, int j) {
		int x0 = i%W_;
		int y0 = i / W_;
		int x1 = j%W_;
		int y1 = j / W_;
		return std::max(EPSILON, exp(-(sqr(x0 - x1) + sqr(y0 - y1)) / gamma));
	}
	void log_convolve(const T* log_u, T* result, int nvectors = 1) const {

		for (int i = 0; i<nvectors; i++) {
			Halide::Buffer<T> resbuff((result + i*N), W_, H_);
			Halide::Buffer<T> ubuff(const_cast<T*>(log_u + i*N), W_, H_);
			in->set(ubuff);
			regul->set(gamma);
			gauss.realize(resbuff);
		}
	}


	void log_convolveAdjoint(const T* u, T* result, int nvectors = 1) const {
		log_convolve(u, result, nvectors);
	}

	void log_convolve_signArray(const T* log_u, unsigned char * signArray, T* result, int nvectors = 1) const {

		T* tmp = new T[W_*H_]; // allocating here ; otherwise not thread-safe

		unsigned char * tmpSignArray = new unsigned char[(W_*H_ + 7) / 8];

		int W_2 = 8*(W_/8);
		int W_2r = W_%8;
		int H_2 = 8*(H_/8);
		int H_2r = H_%8;

		for (int nv = 0; nv<nvectors; nv++) {
#pragma omp parallel for
			for (int i = 0; i<H_; i++) {
				for (int j = 0; j<W_; j++) {
					T v_max = -std::numeric_limits<T>::max();
					for (int k = 0; k<W_; k++) {
						v_max = std::max(v_max, -sqr(j - k) / gamma + log_u[nv*N + i*W_ + k]);
					}

					T conv = 0;
					int k;
					for (k = 0; k<W_2; k+=8) {
						int id = nv*N + i*W_ + k;
						unsigned char s = (char)(((((int)signArray[(id/8)+1])<<8) + signArray[id/8])>>(id%8));

						conv += exp(-sqr(j - k  ) / gamma + log_u[id   ] - v_max) * (((s&1)<<1) - 1);
						conv += exp(-sqr(j - k-1) / gamma + log_u[id +1] - v_max) * ((s & 2) - 1);
						conv += exp(-sqr(j - k-2) / gamma + log_u[id +2] - v_max) * (((s & 4) >> 1) - 1);
						conv += exp(-sqr(j - k-3) / gamma + log_u[id +3] - v_max) * (((s & 8) >> 2) - 1);
						conv += exp(-sqr(j - k-4) / gamma + log_u[id +4] - v_max) * (((s & 16) >> 3) - 1);
						conv += exp(-sqr(j - k-5) / gamma + log_u[id +5] - v_max) * (((s & 32) >> 4) - 1);
						conv += exp(-sqr(j - k-6) / gamma + log_u[id +6] - v_max) * (((s & 64) >> 5) - 1);
						conv += exp(-sqr(j - k-7) / gamma + log_u[id +7] - v_max) * (((s & 128) >> 6) - 1);
					}
					if(W_2r > 0) {
						int id = nv*N + i*W_ + k;
						unsigned char s = (char)(((((int)signArray[(id/8)+1])<<8) + signArray[id/8])>>(id%8));
						for(int b = 0; b < W_2r ; b++)
						{
							conv += exp(-sqr(j - k-b) / gamma + log_u[id + b] - v_max) * ((((s>>b)&1)<<1) - 1);
						}
					}

					tmp[i + j*H_] = myAbsLog(conv, tmpSignArray, i + j*H_) + v_max;
				}
			}

#pragma omp parallel for
			for (int j = 0; j<W_; j++) {
				for (int i = 0; i<H_; i++) {
					T v_max = -std::numeric_limits<T>::max();
					for (int k = 0; k<H_; k++) {
						v_max = std::max(v_max, -sqr(i - k) / gamma + tmp[k + j*H_]);
					}

					T conv = 0;
					int k;
					for (k = 0; k<H_2; k+=8) {
						int id = j*H_ + k;
						unsigned char s = (char)(((((int)tmpSignArray[(id/8)+1])<<8) + tmpSignArray[id/8])>>(id%8));

						conv += exp(-sqr(i - k  ) / gamma + tmp[id   ] - v_max) * (((s & 1) << 1) - 1);
						conv += exp(-sqr(i - k-1) / gamma + tmp[id +1] - v_max) * ((s & 2) - 1);
						conv += exp(-sqr(i - k-2) / gamma + tmp[id +2] - v_max) * (((s & 4) >> 1) - 1);
						conv += exp(-sqr(i - k-3) / gamma + tmp[id +3] - v_max) * (((s & 8) >> 2) - 1);
						conv += exp(-sqr(i - k-4) / gamma + tmp[id +4] - v_max) * (((s & 16) >> 3) - 1);
						conv += exp(-sqr(i - k-5) / gamma + tmp[id +5] - v_max) * (((s & 32) >> 4) - 1);
						conv += exp(-sqr(i - k-6) / gamma + tmp[id +6] - v_max) * (((s & 64) >> 5) - 1);
						conv += exp(-sqr(i - k-7) / gamma + tmp[id +7] - v_max) * (((s & 128) >> 6) - 1);
					}
					if(H_2r > 0) {
						int id = j*H_ + k;
						unsigned char s = (char)(((((int)tmpSignArray[(id/8)+1])<<8) + tmpSignArray[id/8])>>(id%8));
						for(int b = 0; b < H_2r ; b++)
						{
							conv += exp(-sqr(i - k-b) / gamma + tmp[id + b] - v_max) * ((((s>>b)&1)<<1) - 1);
						}
					}

					result[nv*N + i*W_ + j] = myAbsLog(conv, signArray, nv*N + i*W_ + j) + v_max;
				}
			}
		}

		delete[] tmp;
		delete[] tmpSignArray;
	}
	void log_convolve_signArrayAdjoint(const T* u, unsigned char * sign_array, T* result, int nvectors = 1) const {
		log_convolve_signArray(u, sign_array, result, nvectors);
	}

	// Kept here so that it compiles. Shouldn't be used
	void convolve(const T* u, T* result, int nvectors = 1) const {

		T* tmp = new T[W_*H_]; // allocating here ; otherwise not thread-safe

		for (int nv = 0; nv<nvectors; nv++) {

#pragma omp parallel for
			for (int i = 0; i<H_; i++) {
				for (int j = 0; j<W_; j++) {
					T conv = 0;
					for (int k = 0; k<W_; k++) {
						conv += kernel1d[abs(j - k)] * u[nv*N + i*W_ + k];
					}
					tmp[i + j*H_] = conv;
				}
			}

#pragma omp parallel for
			for (int j = 0; j<W_; j++) {
				for (int i = 0; i<H_; i++) {
					T conv = 0;
					for (int k = 0; k<H_; k++) {
						conv += kernel1d[abs(i - k)] * tmp[k + j*H_];
					}
					result[nv*N + i*W_ + j] = conv;
				}
			}
		}
		delete[] tmp;
	}
	// Kept here so that it compiles. Shouldn't be used
	void convolveAdjoint(const T* u, T* result, int nvectors = 1) const {
		convolve(u, result, nvectors);
	}

	ImageParam *in, *signs;
	Param<T>* regul;
	Param<int>* id;
	mutable Func gauss;
	mutable Func gauss_sign;
	T gamma;
	size_t W_, H_;
	size_t N;
	T* kernel1d;
};

#endif
