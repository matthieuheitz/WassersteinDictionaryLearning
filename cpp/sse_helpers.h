/// author: Nicolas Bonneel (nbonneel@seas.harvard.edu)
// small helper for SSE and AVX

#pragma once

#include <stdlib.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#if defined(__linux__) && !defined(_ISOC11_SOURCE)
    #include <malloc.h>
#endif

#include <cstddef>

#ifndef _MSC_VER
#include <iostream>
#include <limits>
#endif

#ifdef AVX_SUPPORT
#define SSE_SUPPORT
#endif

static inline void * malloc_simd(const size_t size, const size_t alignment)
{
#if defined WIN32           // WIN32
    return _aligned_malloc(size, alignment);
#elif defined __linux__     // Linux
    #if defined _ISOC11_SOURCE
	    return aligned_alloc(alignment, size);
    #else
	    return memalign(alignment, size);
    #endif
#elif defined __MACH__      // Mac OS X
    return malloc(size);
#else                       // other (use valloc for page-aligned memory)
    return valloc(size);
#endif
}

static inline void free_simd(void* mem)
{
#if defined WIN32           // WIN32
    return _aligned_free(mem);
#elif defined __linux__     // Linux
    free(mem);
#elif defined __MACH__      // Mac OS X
    free(mem);
#else                       // other (use valloc for page-aligned memory)
    free(mem);
#endif
}




#if _MSC_VER
///grrr... I don't have it ! if your compiler has it and complains, just remove this approximation
static const __m256d ct_acos_1 = _mm256_set1_pd(1.43);
static const __m256d ct_acos_2 = _mm256_set1_pd(0.59);
static const __m256d ct_acos_3 = _mm256_set1_pd(1.65);
static const __m256d ct_acos_4 = _mm256_set1_pd(-1.41);
static const __m256d ct_acos_5 = _mm256_set1_pd(0.88);
static const __m256d ct_acos_6 = _mm256_set1_pd(-0.77);
static const __m256d ct_acos_7 = _mm256_set1_pd(8./3.);
static const __m256d ct_acos_8 = _mm256_set1_pd(-1./3.);
static const __m256d ct_acos_two = _mm256_set1_pd(2.);
static const __m256d ct_acos_mtwo = _mm256_set1_pd(-2.);
static const __m256d ct_acos_half = _mm256_set1_pd(0.5);
static const __m256d ct_acos_invsix = _mm256_set1_pd(1./6.);
static const __m256d ct_acos_eight = _mm256_set1_pd(8.);

static inline __m256d _mm256_acos_pd(__m256d x) {
	return _mm256_set_pd(acos(x.m256d_f64[0]), acos(x.m256d_f64[1]), acos(x.m256d_f64[2]), acos(x.m256d_f64[3]));
	// approximation below not precise enough

	__m256d a = _mm256_add_pd(ct_acos_1, _mm256_mul_pd(ct_acos_2, x));
	a = _mm256_mul_pd(_mm256_add_pd(a, _mm256_div_pd(_mm256_add_pd(ct_acos_two, _mm256_mul_pd(ct_acos_two, x)), a)), ct_acos_half);
	__m256d b = _mm256_add_pd(ct_acos_3, _mm256_mul_pd(ct_acos_4, x));
	b = _mm256_mul_pd(_mm256_add_pd(b, _mm256_div_pd(_mm256_add_pd(ct_acos_two, _mm256_mul_pd(ct_acos_mtwo, x)), b)), ct_acos_half);
	__m256d c = _mm256_add_pd(ct_acos_5, _mm256_mul_pd(ct_acos_6, x));
	c = _mm256_mul_pd(_mm256_add_pd(c, _mm256_div_pd(_mm256_sub_pd(ct_acos_two, a), c)), ct_acos_half);
	return _mm256_mul_pd(_mm256_sub_pd(_mm256_mul_pd(ct_acos_eight, _mm256_add_pd(c, _mm256_div_pd(_mm256_sub_pd(ct_acos_two, a), c))), _mm256_add_pd(b, _mm256_div_pd(_mm256_add_pd(ct_acos_two, _mm256_mul_pd(ct_acos_mtwo, x)), b))), ct_acos_invsix);
}

static const __m256d _ps256_exp_hi = _mm256_set1_pd(200.3762626647949f);
static const __m256d _ps256_exp_lo = _mm256_set1_pd(-200.3762626647949f);

static const __m256d _ps256_cephes_LOG2EF = _mm256_set1_pd(1.44269504088896341);
static const __m256d _ps256_cephes_exp_C1 = _mm256_set1_pd(0.693359375);
static const __m256d _ps256_cephes_exp_C2 = _mm256_set1_pd(-2.12194440e-4);

static const __m256d _ps256_cephes_exp_p0 = _mm256_set1_pd(1.9875691500E-4);
static const __m256d _ps256_cephes_exp_p1 = _mm256_set1_pd(1.3981999507E-3);
static const __m256d _ps256_cephes_exp_p2 = _mm256_set1_pd(8.3334519073E-3);
static const __m256d _ps256_cephes_exp_p3 = _mm256_set1_pd(4.1665795894E-2);
static const __m256d _ps256_cephes_exp_p4 = _mm256_set1_pd(1.6666665459E-1);
static const __m256d _ps256_cephes_exp_p5 = _mm256_set1_pd(5.0000001201E-1);
static const __m256d _ps256_exp_0p5 = _mm256_set1_pd(0.5);

static const __m128i _pi32_256_123 = _mm_set1_epi32(1023);


static inline __m256d _mm256_exp_pd(__m256d x) {

	return _mm256_set_pd(exp(x.m256d_f64[0]), exp(x.m256d_f64[1]), exp(x.m256d_f64[2]), exp(x.m256d_f64[3]));
	// approximation below not precise enough

	__m256d tmp = _mm256_setzero_pd(), fx;
	__m128i imm0;
	__m256d one = _mm256_set1_pd(1.);

	x = _mm256_min_pd(x, _ps256_exp_hi);
	x = _mm256_max_pd(x, _ps256_exp_lo);

	/* express exp(x) as exp(g + n*log(2)) */
	fx = _mm256_mul_pd(x, _ps256_cephes_LOG2EF);
	fx = _mm256_add_pd(fx, _ps256_exp_0p5);

	/* how to perform a floorf with SSE: just below */
	//imm0 = _mm256_cvttps_epi32(fx);
	//tmp  = _mm256_cvtepi32_ps(imm0);

	tmp = _mm256_floor_pd(fx);

	/* if greater, substract 1 */
	//v8sf mask = _mm256_cmpgt_ps(tmp, fx);
	__m256d mask = _mm256_cmp_pd(tmp, fx, _CMP_GT_OS);
	mask = _mm256_and_pd(mask, one);
	fx = _mm256_sub_pd(tmp, mask);

	tmp = _mm256_mul_pd(fx, _ps256_cephes_exp_C1);
	__m256d z = _mm256_mul_pd(fx, _ps256_cephes_exp_C2);
	x = _mm256_sub_pd(x, tmp);
	x = _mm256_sub_pd(x, z);

	z = _mm256_mul_pd(x, x);

	__m256d y = _ps256_cephes_exp_p0;
	y = _mm256_mul_pd(y, x);
	y = _mm256_add_pd(y, _ps256_cephes_exp_p1);
	y = _mm256_mul_pd(y, x);
	y = _mm256_add_pd(y, _ps256_cephes_exp_p2);
	y = _mm256_mul_pd(y, x);
	y = _mm256_add_pd(y, _ps256_cephes_exp_p3);
	y = _mm256_mul_pd(y, x);
	y = _mm256_add_pd(y, _ps256_cephes_exp_p4);
	y = _mm256_mul_pd(y, x);
	y = _mm256_add_pd(y, _ps256_cephes_exp_p5);
	y = _mm256_mul_pd(y, z);
	y = _mm256_add_pd(y, x);
	y = _mm256_add_pd(y, one);

	/* build 2^n */
	//__m128 d1 = _mm128_cvttps_epi32(fx);
	imm0 = _mm256_cvttpd_epi32(fx);
	// another two AVX2 instructions
	imm0 = _mm_add_epi32(imm0, _pi32_256_123);

	/*__m256i ik;
	ik.m256i_i64[0] = imm0.m128i_i32[0];
	ik.m256i_i64[1] = imm0.m128i_i32[1];
	ik.m256i_i64[2] = imm0.m128i_i32[2];
	ik.m256i_i64[3] = imm0.m128i_i32[3];

	ik = _mm256_slli_epi64(ik, 52);

	__m256d pow2n = _mm256_castsi256_pd(ik);
	y = _mm256_mul_pd(y, pow2n);*/

	__m256d pow2d; // my proc doesn't like _mm256_slli_epi64, my compiler doesn't have _mm_add_epi64, etc. etc. etc. ! grrrr
	unsigned long long ul1 = ((unsigned long long) imm0.m128i_u32[0]) << 52l;
	unsigned long long ul2 = ((unsigned long long) imm0.m128i_u32[1]) << 52l;
	unsigned long long ul3 = ((unsigned long long) imm0.m128i_u32[2]) << 52l;
	unsigned long long ul4 = ((unsigned long long) imm0.m128i_u32[3]) << 52l;
	pow2d.m256d_f64[0] = *((double*)&ul1);
	pow2d.m256d_f64[1] = *((double*)&ul2);
	pow2d.m256d_f64[2] = *((double*)&ul3);
	pow2d.m256d_f64[3] = *((double*)&ul4);

	/*long long int t = 1025ll<<52ll;
	double d = *((double*)&t);*/

	return _mm256_mul_pd(y, pow2d);

}


#else

// emulates Visual Studio's m256d_f64 fields, and missing SSE/AVX instructions
union m256d {
	struct { double m256d_f64[4]; };
	__m256d variable;
};

static inline __m256d _mm256_acos_pd(__m256d y) {
	m256d x; x.variable = y;
	return _mm256_set_pd(acos(x.m256d_f64[0]), acos(x.m256d_f64[1]), acos(x.m256d_f64[2]), acos(x.m256d_f64[3]));
	// tried approximate acos, but not precise enough
}
static inline __m256d _mm256_exp_pd(__m256d y) {
	m256d x; x.variable = y;
	return _mm256_set_pd(exp(x.m256d_f64[0]), exp(x.m256d_f64[1]), exp(x.m256d_f64[2]), exp(x.m256d_f64[3]));
	// tried approximate exp, but not precise enough
}

#endif

#ifdef SSE_SUPPORT

#ifdef AVX_SUPPORT
#define ALIGN 32
#define VECSIZEDOUBLE 4
#define VECSIZEFLOAT 8
//#define simd_dotp(x,y) _mm256_dp_pd(x,y)
#define simd_add(x,y) _mm256_add_pd(x,y)
#define simd_sub(x,y) _mm256_sub_pd(x,y)
#define simd_mul(x,y) _mm256_mul_pd(x,y)
#define simd_div(x,y) _mm256_div_pd(x,y)
#define simd_max(x,y) _mm256_max_pd(x,y)
#define simd_load(x) _mm256_load_pd(x)
#define simd_store(x,y) _mm256_store_pd(x,y)
#define simd_set1(x) _mm256_set1_pd(x)
#define simd_or(x,y) _mm256_or_pd(x,y)
#define simd_gt(x,y) _mm256_cmp_pd(x,y,_CMP_GT_OS)
#define simd_and(x,y) _mm256_and_pd(x,y)
#define simd_andnot(x,y) _mm256_andnot_pd(x,y)

#define simd_dotp_f(x,y) _mm256_dp_ps(x,y)
#define simd_add_f(x,y) _mm256_add_ps(x,y)
#define simd_sub_f(x,y) _mm256_sub_ps(x,y)
#define simd_mul_f(x,y) _mm256_mul_ps(x,y)
#define simd_div_f(x,y) _mm256_div_ps(x,y)
#define simd_max_f(x,y) _mm256_max_ps(x,y)
#define simd_load_f(x) _mm256_load_ps(x)
#define simd_store_f(x,y) _mm256_store_ps(x,y)
#define simd_set1_f(x) _mm256_set1_ps(x)
#define simd_or_f(x,y) _mm256_or_ps(x,y)
#define simd_gt_f(x,y) _mm256_cmp_ps(x,y,_CMP_GT_OS)
#define simd_and_f(x,y) _mm256_and_ps(x,y)
#define simd_andnot_f(x,y) _mm256_andnot_ps(x,y)

typedef __m256d simd_double;
typedef __m256 simd_float;



static inline double dotp16(const double* u, const double* v) {
	__m256d xy0 = _mm256_mul_pd(simd_load(u), simd_load(v));
	__m256d xy1 = _mm256_mul_pd(simd_load(u+4), simd_load(v+4));
	__m256d xy2 = _mm256_mul_pd(simd_load(u+8), simd_load(v+8));
	__m256d xy3 = _mm256_mul_pd(simd_load(u+12), simd_load(v+12));

	__m256d dotproduct = _mm256_add_pd(_mm256_add_pd(xy0, xy1), _mm256_add_pd(xy2, xy3));
#ifdef _MSC_VER
	__m256d s = _mm256_hadd_pd(dotproduct, dotproduct);
	return s.m256d_f64[0]+s.m256d_f64[2];
#else
	m256d s;
	s.variable = _mm256_hadd_pd(dotproduct, dotproduct);
	return s.m256d_f64[0]+s.m256d_f64[2];
#endif
}

static inline double dotp32(const double* u, const double* v) {
	__m256d xy0 = _mm256_mul_pd(simd_load(u), simd_load(v));
	__m256d xy1 = _mm256_mul_pd(simd_load(u+4), simd_load(v+4));
	__m256d xy2 = _mm256_mul_pd(simd_load(u+8), simd_load(v+8));
	__m256d xy3 = _mm256_mul_pd(simd_load(u+12), simd_load(v+12));

	__m256d xy4 = _mm256_mul_pd(simd_load(u+16), simd_load(v+16));
	__m256d xy5 = _mm256_mul_pd(simd_load(u+20), simd_load(v+20));
	__m256d xy6 = _mm256_mul_pd(simd_load(u+24), simd_load(v+24));
	__m256d xy7 = _mm256_mul_pd(simd_load(u+28), simd_load(v+28));

	__m256d dotproduct1 = _mm256_add_pd(_mm256_add_pd(xy0, xy1), _mm256_add_pd(xy2, xy3));
	__m256d dotproduct2 = _mm256_add_pd(_mm256_add_pd(xy4, xy5), _mm256_add_pd(xy6, xy7));
	__m256d dotproduct = _mm256_add_pd(dotproduct1, dotproduct2);
#ifdef _MSC_VER
	__m256d s = _mm256_hadd_pd(dotproduct, dotproduct);
	return s.m256d_f64[0]+s.m256d_f64[2];
#else
	m256d s;
	s.variable = _mm256_hadd_pd(dotproduct, dotproduct);
	return s.m256d_f64[0]+s.m256d_f64[2];
#endif
}

static inline double dotp64(const double* u, const double* v) {
	__m256d xy0 = _mm256_mul_pd(simd_load(u), simd_load(v));
	__m256d xy1 = _mm256_mul_pd(simd_load(u+4), simd_load(v+4));
	__m256d xy2 = _mm256_mul_pd(simd_load(u+8), simd_load(v+8));
	__m256d xy3 = _mm256_mul_pd(simd_load(u+12), simd_load(v+12));

	__m256d xy4 = _mm256_mul_pd(simd_load(u+16), simd_load(v+16));
	__m256d xy5 = _mm256_mul_pd(simd_load(u+20), simd_load(v+20));
	__m256d xy6 = _mm256_mul_pd(simd_load(u+24), simd_load(v+24));
	__m256d xy7 = _mm256_mul_pd(simd_load(u+28), simd_load(v+28));

	__m256d xy8 = _mm256_mul_pd(simd_load(u+32), simd_load(v+32));
	__m256d xy9 = _mm256_mul_pd(simd_load(u+36), simd_load(v+36));
	__m256d xy10 = _mm256_mul_pd(simd_load(u+40), simd_load(v+40));
	__m256d xy11 = _mm256_mul_pd(simd_load(u+44), simd_load(v+44));

	__m256d xy12 = _mm256_mul_pd(simd_load(u+48), simd_load(v+48));
	__m256d xy13 = _mm256_mul_pd(simd_load(u+52), simd_load(v+52));
	__m256d xy14 = _mm256_mul_pd(simd_load(u+56), simd_load(v+56));
	__m256d xy15 = _mm256_mul_pd(simd_load(u+60), simd_load(v+60));

	__m256d dotproduct1 = _mm256_add_pd(_mm256_add_pd(xy0, xy1), _mm256_add_pd(xy2, xy3));
	__m256d dotproduct2 = _mm256_add_pd(_mm256_add_pd(xy4, xy5), _mm256_add_pd(xy6, xy7));
	__m256d dotproduct3 = _mm256_add_pd(_mm256_add_pd(xy8, xy9), _mm256_add_pd(xy10, xy11));
	__m256d dotproduct4 = _mm256_add_pd(_mm256_add_pd(xy12, xy13), _mm256_add_pd(xy14, xy15));

	__m256d dotproductA = _mm256_add_pd(dotproduct1, dotproduct2);
	__m256d dotproductB = _mm256_add_pd(dotproduct3, dotproduct4);

	__m256d dotproduct = _mm256_add_pd(dotproductA, dotproductB);
#ifdef _MSC_VER
	__m256d s = _mm256_hadd_pd(dotproduct, dotproduct);
	return s.m256d_f64[0]+s.m256d_f64[2];
#else
	m256d s;
	s.variable = _mm256_hadd_pd(dotproduct, dotproduct);
	return s.m256d_f64[0]+s.m256d_f64[2];
#endif
}

static inline double dotp128(const double* u, const double* v) {
	__m256d xy0 = _mm256_mul_pd(simd_load(u), simd_load(v));
	__m256d xy1 = _mm256_mul_pd(simd_load(u+4), simd_load(v+4));
	__m256d xy2 = _mm256_mul_pd(simd_load(u+8), simd_load(v+8));
	__m256d xy3 = _mm256_mul_pd(simd_load(u+12), simd_load(v+12));

	__m256d xy4 = _mm256_mul_pd(simd_load(u+16), simd_load(v+16));
	__m256d xy5 = _mm256_mul_pd(simd_load(u+20), simd_load(v+20));
	__m256d xy6 = _mm256_mul_pd(simd_load(u+24), simd_load(v+24));
	__m256d xy7 = _mm256_mul_pd(simd_load(u+28), simd_load(v+28));

	__m256d xy8 = _mm256_mul_pd(simd_load(u+32), simd_load(v+32));
	__m256d xy9 = _mm256_mul_pd(simd_load(u+36), simd_load(v+36));
	__m256d xy10 = _mm256_mul_pd(simd_load(u+40), simd_load(v+40));
	__m256d xy11 = _mm256_mul_pd(simd_load(u+44), simd_load(v+44));

	__m256d xy12 = _mm256_mul_pd(simd_load(u+48), simd_load(v+48));
	__m256d xy13 = _mm256_mul_pd(simd_load(u+52), simd_load(v+52));
	__m256d xy14 = _mm256_mul_pd(simd_load(u+56), simd_load(v+56));
	__m256d xy15 = _mm256_mul_pd(simd_load(u+60), simd_load(v+60));

	__m256d xy16 = _mm256_mul_pd(simd_load(u+64), simd_load(v+64));
	__m256d xy17 = _mm256_mul_pd(simd_load(u+68), simd_load(v+68));
	__m256d xy18 = _mm256_mul_pd(simd_load(u+72), simd_load(v+72));
	__m256d xy19 = _mm256_mul_pd(simd_load(u+76), simd_load(v+76));

	__m256d xy20 = _mm256_mul_pd(simd_load(u+80), simd_load(v+80));
	__m256d xy21 = _mm256_mul_pd(simd_load(u+84), simd_load(v+84));
	__m256d xy22 = _mm256_mul_pd(simd_load(u+88), simd_load(v+88));
	__m256d xy23 = _mm256_mul_pd(simd_load(u+92), simd_load(v+92));

	__m256d xy24 = _mm256_mul_pd(simd_load(u+96), simd_load(v+96));
	__m256d xy25 = _mm256_mul_pd(simd_load(u+100), simd_load(v+100));
	__m256d xy26 = _mm256_mul_pd(simd_load(u+104), simd_load(v+104));
	__m256d xy27 = _mm256_mul_pd(simd_load(u+108), simd_load(v+108));

	__m256d xy28 = _mm256_mul_pd(simd_load(u+112), simd_load(v+112));
	__m256d xy29 = _mm256_mul_pd(simd_load(u+116), simd_load(v+116));
	__m256d xy30 = _mm256_mul_pd(simd_load(u+120), simd_load(v+120));
	__m256d xy31 = _mm256_mul_pd(simd_load(u+124), simd_load(v+124));

	__m256d dotproduct1a = _mm256_add_pd(_mm256_add_pd(xy0, xy1), _mm256_add_pd(xy2, xy3));
	__m256d dotproduct2a = _mm256_add_pd(_mm256_add_pd(xy4, xy5), _mm256_add_pd(xy6, xy7));
	__m256d dotproduct3a = _mm256_add_pd(_mm256_add_pd(xy8, xy9), _mm256_add_pd(xy10, xy11));
	__m256d dotproduct4a = _mm256_add_pd(_mm256_add_pd(xy12, xy13), _mm256_add_pd(xy14, xy15));

	__m256d dotproduct1b = _mm256_add_pd(_mm256_add_pd(xy16, xy17), _mm256_add_pd(xy18, xy19));
	__m256d dotproduct2b = _mm256_add_pd(_mm256_add_pd(xy20, xy21), _mm256_add_pd(xy22, xy23));
	__m256d dotproduct3b = _mm256_add_pd(_mm256_add_pd(xy24, xy25), _mm256_add_pd(xy26, xy27));
	__m256d dotproduct4b = _mm256_add_pd(_mm256_add_pd(xy28, xy29), _mm256_add_pd(xy30, xy31));


	__m256d dotproductA = _mm256_add_pd(dotproduct1a, dotproduct2a);
	__m256d dotproductB = _mm256_add_pd(dotproduct3a, dotproduct4a);
	__m256d dotproductC = _mm256_add_pd(dotproduct1b, dotproduct2b);
	__m256d dotproductD = _mm256_add_pd(dotproduct3b, dotproduct4b);


	__m256d dotproduct1 = _mm256_add_pd(dotproductA, dotproductB);
	__m256d dotproduct2 = _mm256_add_pd(dotproductC, dotproductD);

	__m256d dotproduct = _mm256_add_pd(dotproduct1, dotproduct2);
#ifdef _MSC_VER
	__m256d s = _mm256_hadd_pd(dotproduct, dotproduct);
	return s.m256d_f64[0]+s.m256d_f64[2];
#else
	m256d s;
	s.variable = _mm256_hadd_pd(dotproduct, dotproduct);
	return s.m256d_f64[0]+s.m256d_f64[2];
#endif
}

static inline double dotp_full(const double* u, const double* v, int n) {

	int k=0;
	const int max128 = n-128+1;
	const int max64 = n-64+1;
	const int max16 = n-16+1;
	double conv = 0;
	for (; k<max128; k+=128) {
		conv += dotp128(&u[k], &v[k]);
	}
	for (; k<max64; k+=64) {
		conv += dotp64(&u[k], &v[k]);
	}
	for (; k<max16; k+=16) {
		conv += dotp16(&u[k], &v[k]);
	}
	for (; k<n; k++) {
		conv += u[k]*v[k];
	}
	return conv;
}

#else

#define ALIGN 16
#define VECSIZEDOUBLE 2
#define VECSIZEFLOAT 4
#define simd_dotp(x,y) _mm_dp_pd(x,y)
#define simd_add(x,y) _mm_add_pd(x,y)
#define simd_sub(x,y) _mm_sub_pd(x,y)
#define simd_mul(x,y) _mm_mul_pd(x,y)
#define simd_div(x,y) _mm_div_pd(x,y)
#define simd_max(x,y) _mm_max_pd(x,y)
#define simd_load(x) _mm_load_pd(x)
#define simd_store(x,y) _mm_store_pd(x,y)
#define simd_set1(x) _mm_set1_pd(x)
#define simd_or(x,y) _mm_or_pd(x,y)
#define simd_gt(x,y) _mm_cmpgt_pd(x,y)
#define simd_and(x,y) _mm_and_pd(x,y)
#define simd_andnot(x,y) _mm_andnot_pd(x,y)

#define simd_dotp_f(x,y) _mm_dp_ps(x,y)
#define simd_add_f(x,y) _mm_add_ps(x,y)
#define simd_sub_f(x,y) _mm_sub_ps(x,y)
#define simd_mul_f(x,y) _mm_mul_ps(x,y)
#define simd_div_f(x,y) _mm_div_ps(x,y)
#define simd_max_f(x,y) _mm_max_ps(x,y)
#define simd_load_f(x) _mm_load_ps(x)
#define simd_store_f(x,y) _mm_store_ps(x,y)
#define simd_set1_f(x) _mm_set1_ps(x)
#define simd_or_f(x,y) _mm_or_ps(x,y)
#define simd_gt_f(x,y) _mm_cmpgt_ps(x,y)
#define simd_and_f(x,y) _mm_and_ps(x,y)
#define simd_andnot_f(x,y) _mm_andnot_ps(x,y)

typedef __m128d simd_double;
typedef __m128 simd_float;
#endif

#endif


// to align std::vector
// http://www.gamedev.net/topic/391394-stl-and-aligned-memory/
template <class T, int Alignment=ALIGN>
class aligned_allocator {
public:

	typedef size_t    size_type;
	typedef ptrdiff_t difference_type;
	typedef T*        pointer;
	typedef const T*  const_pointer;
	typedef T&        reference;
	typedef const T&  const_reference;
	typedef T         value_type;


	template <class U>
	struct rebind {
		typedef aligned_allocator<U> other;
	};


	pointer address(reference value) const {
		return &value;
	};

	const_pointer address(const_reference value) const {
		return &value;
	};


	aligned_allocator() throw() {
	};

	aligned_allocator(const aligned_allocator&) throw() {
	};

	template <class U>
	aligned_allocator(const aligned_allocator<U>&) throw() {
	};

	~aligned_allocator() throw() {
	};

	//max capacity
	size_type max_size() const throw() {
		return std::numeric_limits<size_type>::max();
	};


	pointer allocate(size_type num, const_pointer *hint = 0) {

		return (pointer)malloc_simd(num*sizeof(T), Alignment);
	};


	void construct(pointer p, const T& value) {

		// memcpy( p, &value, sizeof T );
		*p=value;
		//  new ( (void *) p ) T ( value );
	};


	void destroy(pointer p) {

		p->~T();
	};


	void deallocate(pointer p, size_type num) {

		free_simd(p);
	};
};
