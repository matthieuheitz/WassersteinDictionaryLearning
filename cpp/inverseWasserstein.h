#pragma once
#include <vector>
#include "lbfgs.h"
#include "kernels.h"
#include "loss.h"
#include "histogramIO.h"
#include "signArray.h"
#include "chrono.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <complex>

typedef std::complex<float> ComplexDataType;

enum gradient_type {GRADIENT_IMPLICIT, GRADIENT_SINKHORN, GRADIENT_NUMERIC};

template<class KernelType> class Problem;
template<class KernelType> class WassersteinRegression;
template<class KernelType> class WassersteinBarycenter;

template<typename T> T dot(const T* u, const T* v, int N) {
	double res = 0.;
	for (int i=0; i<N; i++) res += u[i]*v[i];
	return res;
}


template<class KernelType>
class Problem {
public:
	Problem(std::vector<std::vector<double> > &histograms, const std::vector<double> &observed_histogram, const KernelType &pkernel, double* weights = NULL, int npdfs = 1) {
		pdfs = histograms;
		K = pdfs.size();
		N = pkernel.N;
		observed_pdf = observed_histogram;
		kernel = pkernel;
		lambdas = new double[K];
		num_pdfs = npdfs;
		if (weights) { set_weights(weights); };

	}
	Problem(const Problem& p) {  // Kernel is passed by pointer ; lambdas are copied
		pdfs = p.pdfs;
		K = p.K;
		N = p.N;
		observed_pdf = p.observed_pdf;
		kernel = p.kernel;
		num_pdfs = p.num_pdfs;
		lambdas = new double[K];
		set_weights(p.lambdas);
	}
	void normalize_values() {

		int P = num_pdfs;

		double s = 0;
		for (int i=0; i<K; i++) {
			s+=lambdas[i];
		}
		for (int i=0; i<K; i++) {
			lambdas[i] /= s;
		}
		double s2;
		for (int i=0; i<K; i++) {
			s2=0;
			for (int j=0; j<N; j++) {
				s2+=pdfs[i][j];
			}
			for (int j=0; j<N; j++) {
				pdfs[i][j]/=s2;
			}
		}
	}
	~Problem() {
		delete[] lambdas;
	}
	void set_weights(const double* weights) {
		memcpy(lambdas, weights, K*sizeof(double));
	}

	void set_pdfs(const double* vertex_pdfs) {
		for(int i=0; i<K; ++i)
		{
			pdfs[i].assign(vertex_pdfs+i*N,vertex_pdfs+(i+1)*N);
		}
	}

	std::vector<std::vector<double> > pdfs;
	std::vector<double> observed_pdf;
	KernelType kernel;
	double* lambdas;
	size_t N, K;
	int num_pdfs;
};



template<class KernelType>
class WassersteinBarycenter {
	template<class T> friend class WassersteinRegression;

public:
	WassersteinBarycenter(Problem<KernelType>* p, int n_bregman_iter) {
		K = p->pdfs.size();
		N = p->kernel.N;
		problem = p;
		Niters = n_bregman_iter;
	}

	double get_plan(int id, int i, int j) {
		return a[id*N+i]*b[id*N+j]*problem->kernel(i, j);
	}

	void compute_barycenter(int id_basis = 0) {
		b.resize(K*N);
		std::fill(b.begin(), b.begin()+K*N, 1.0);
		compute_barycenter_no_scaling_init(id_basis);
	}

	// Bregman projections
	void compute_barycenter_no_scaling_init(int id_basis = 0) {

		barycenter.resize(N);
		a.resize(K*N);

		std::vector<double, aligned_allocator<double> > convolution(K*N);

		problem->normalize_values();
		// Bregman Projections
		for (int iter=0; iter<Niters; iter++) {

//#pragma omp parallel for
			problem->kernel.convolve(&b[0], &convolution[0], K);
			for (int i=0; i<K; i++) {
				for (int j=0; j<N; j++) {
					a[i*N+j] = problem->pdfs[i][id_basis*N + j] / convolution[i*N+j];
				}
			}
			problem->kernel.convolveAdjoint(&a[0], &convolution[0], K);

			geomMean(&convolution[0], &barycenter[0]);

//#pragma omp parallel for
			for (int i=0; i<K; i++) {
				for (int j=0; j<N; j++) {
					b[i*N+j] = barycenter[j] / convolution[i*N+j];
				}

			}

		}
	}

	void compute_barycenter_log(int id_basis = 0) {

		barycenter.resize(N);

		a.resize(K*N);
		b.resize(K*N);
		std::fill(b.begin(), b.begin()+K*N, 0.0);

		std::vector<double, aligned_allocator<double> > convolution(K*N);

		problem->normalize_values();
		// Bregman Projections
		for (int iter=0; iter<Niters; iter++) {

//#pragma omp parallel for
//			problem->kernel.convolve(&b[0], &convolution[0], K);
			problem->kernel.log_convolve(&b[0], &convolution[0], K);
			for (int i=0; i<K; i++) {
				for (int j=0; j<N; j++) {
//					a[i*N+j] = problem->pdfs[i][id_basis*N + j] / convolution[i*N+j];
					a[i*N+j] = log(problem->pdfs[i][id_basis*N + j]) - convolution[i*N+j];
				}
			}
//			problem->kernel.convolveAdjoint(&a[0], &convolution[0], K);
			problem->kernel.log_convolveAdjoint(&a[0], &convolution[0], K);

			memset(&barycenter[0], 0, N*sizeof(barycenter[0]));
			for (int i=0; i<K; i++) {
				double l = problem->lambdas[i];
				for (int j=0; j<N; j++) {
	//				barycenter[j] += safe_log(phi[iter*K*N + i*N+j])*l;
					barycenter[j] += convolution[i*N+j]*l;
				}
			}

#pragma omp parallel for
			for (int i=0; i<K; i++) {
				for (int j=0; j<N; j++) {
//					b[i*N+j] = barycenter[j] / convolution[i*N+j];
					b[i*N+j] = barycenter[j] - convolution[i*N+j];
				}
			}
		}
		for (int j=0; j<N; j++) {
			barycenter[j] = exp(barycenter[j]);
		}
	}

	void sharpen_barycenter() {

		double maxEntropy = -1E9;
		for (int i=0; i<K; i++) {
			maxEntropy = std::max(maxEntropy, entropy(&problem->pdfs[i][0]));
		}

		double curEntropy = entropy(&barycenter[0]);
		double sumBary = 0; for (int i=0; i<N; i++) sumBary+= barycenter[i];
		if (curEntropy + sumBary > maxEntropy + 1) {


			std::vector<double> baryBeta(N);
			std::vector<double> entrop(40);
#pragma omp parallel for firstprivate(baryBeta)
			for (int i=0; i<entrop.size(); i++) {
				double beta = (i+0.5)/20.;
				for (int j=0; j<N; j++) {
					baryBeta[j] = std::pow(barycenter[j], beta);
				}
				double sumBaryBeta = 0; for (int j=0; j<N; j++) sumBaryBeta+= baryBeta[j];
				double curEntropyBeta = entropy(&baryBeta[0]);
				double s = sqr(sumBaryBeta + curEntropyBeta - (1+maxEntropy));
				entrop[i] = s;
			}

			double bestBeta = 1;
			double bestValue = sqr(sumBary + curEntropy - (1+maxEntropy));
			for (int i=0; i<entrop.size(); i++) {
				double beta = (i+0.5)/20.;
				if (entrop[i] < bestValue) {
					bestValue = entrop[i];
					bestBeta = beta;
				}
			}

			double sum = 0;
			for (int j=0; j<N; j++) {
				barycenter[j] = std::pow(barycenter[j], bestBeta);
				sum += barycenter[j];
			}
			for (int j=0; j<N; j++) {
				barycenter[j] /= sum;
			}
			std::cout<<"Sharpening with beta = "<<bestBeta<<std::endl;
		}

	}

	// Accessors
	const std::vector<double>& get_barycenter() const { return barycenter; }

	// Mutators
	void set_b_init(double * b_init, int size) {
		if(size != K*N)
		{
			std::cerr<<"Init vector for b is too large"<<std::endl;
			return;
		}
		b.assign(b_init, b_init + size);
	}

private:

	double entropy(double* mu) {
		double H = 0;
		for (int i=0; i<N; i++) {
			H+=mu[i]*safe_log(mu[i]);
		}
		return -H;
	}

	// Wasserstein  cost of the transport plan given by a and b
	double cost_W(const std::vector<double> &a, const std::vector<double> &b) const {
		double sum = 0;
		for (int j=0; j<N; j++) {
			for (int k=0; k<N; k++) {
				sum+=problem->kernelMatrix[j*N+k]*a[k]*b[j]*(safe_log(a[k])+safe_log(b[j])-1.);
			}
		}
		return sum;
	}

	// Wasserstein  cost of the transport plan given by the current barycenter
	double cost_W_bary() const {
		double sum = 0;
		for (int i=0; i<K; i++) {
			sum += problem->lambdas[i]*cost_W(a[i], b[i]);
		}
		return sum;
	}

	// geometric mean used for Bregman projections
	void geomMean(double* convolved_a, double* result) const {

		memset(&result[0], 0, N*sizeof(result[0]));
		double sumLambdas = 0;
		for (int i=0; i<K; i++) {
			sumLambdas += problem->lambdas[i];
		}

		for (int i=0; i<K; i++) {
			double l = problem->lambdas[i];
			for (int j=0; j<N; j++) {
				result[j] += safe_log(convolved_a[i*N+j])*l;
			}
		}
		for (int j=0; j<N; j++)
			result[j] = exp(result[j]/*/sumLambdas*/);
	}
public:
	std::vector<double, aligned_allocator<double>  > a, b;
	std::vector<double> barycenter;

	Problem<KernelType> *problem;
	size_t K, N;
	int Niters;
};


template<class KernelType>
class WassersteinRegression {
public:
	WassersteinRegression(Problem<KernelType>* p,
						  int n_bregman_iterations,
						  gradient_type gradient_method,
						  const BaseLoss<KernelType> &lossFunctor,
						  const ExportHistogramBase &exportFunctor,
						  double scale_dictionary = 0.0,					// Only for dual regression (regress_both).
						  bool export_atoms = true,							// Only for dual regression (regress_both)
						  bool export_fittings = true,						// Only for dual regression (regress_both)
						  bool export_only_final_solution = true,
						  bool warm_restart = false) :
			exporter(exportFunctor), loss(lossFunctor)
	{
		K = p->pdfs.size();
		N = p->kernel.N;
		iteration = 0;
		bary_computation = new WassersteinBarycenter<KernelType>(p, n_bregman_iterations);
		problem = p;
		firstCall = true;
		this->gradient_method = gradient_method;
		// Optimization is done in the log-domain (to keep positive values)
		exp_weight = true;
		// Exportation of pdfs
		scaleDictionary = scale_dictionary;
		exportAtoms = export_atoms;
		exportFittings = export_fittings;
		exportOnlyFinalSolution = export_only_final_solution;
		warmRestart = warm_restart;
		exportEveryMIter = 1;
		lbfgs_parameter_init(&lbfgs_param);
	}

	~WassersteinRegression() {
		delete bary_computation;
	}


	void regress_both(double* solution) {
		double residual;
		firstCall = true;
		lbfgs_param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE; // LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE; // LBFGS_LINESEARCH_BACKTRACKING_WOLFE; // LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;// LBFGS_LINESEARCH_MORETHUENTE;
		lbfgs_param.epsilon = 1E-50;		// Convergence test on accuracy
		lbfgs_param.max_linesearch = 20;	// Max number of trials for the line search
		lbfgs_param.delta = 1E-50;		// Convergence test on minimum rate of decrease
		if(warmRestart)
			lbfgs_param.max_iterations = 10; // Number of iterations when doing multiple small runs of LBFGS
		else
			lbfgs_param.max_iterations = wrTotalIteration;

		iteration = 0;

		int P = problem->num_pdfs;
		int n = K*P + K*N;
		double scaleDic = scaleDictionary; // Scale between dictionary and weight sizes

		if(warmRestart)
		{
			b_storage.resize(K*N*P); // Initialize the storage vector for warm restart
			std::fill(b_storage.begin(), b_storage.begin()+K*N*P, 1.0);
			b_temp.resize(K*N*P);
			b.resize(K*N*(this->bary_computation->Niters+1)); // b initialization
			std::fill(b.begin(), b.begin()+K*N, 1.0);
		}

		// Transfer the data to the log domain
		if (exp_weight) {
			for (int i=0; i<P*K; i++) {
				solution[i] = log(solution[i])*scaleDic;
			}
		} else {
			for (int i=0; i<P*K; i++) {
				solution[i] = solution[i]*scaleDic;
			}
		}
		for (int i=0; i<N*K; i++) {
			solution[i+P*K] = log(solution[i+P*K]);
		}

		// Start the timer
		chrono.Start();

		// Run the optimization
		for(int i=0;i<wrTotalIteration/lbfgs_param.max_iterations;++i)
		{
			if(warmRestart)
				std::cout<<std::endl<<"LBGS run # "<<i<<" - Iterations accumulated : "<<i*lbfgs_param.max_iterations<<std::endl<<std::endl;
			// Run the optimization
			int ret = lbfgs(n, solution, &residual, evaluate_both, progress_both, (void*)this, &lbfgs_param);
			if(ret < 0)
			{
				std::cout<<"LBFGS returned code "<<ret<<std::endl;
				std::cout<<lbfgs_strerror(ret)<<std::endl;
			}
			if(this->warmRestart)
			{
				// Store the last computed scalings
				std::copy(this->b_temp.begin(), this->b_temp.end(),this->b_storage.begin());
			}
		}
		std::vector<double> solution2(solution,solution+P*K+K*N);

		// Transfer the data back to the original domain
		if (exp_weight) {
			for (int i=0; i<P*K; i++) {
				solution[i] = exp(solution[i]/scaleDic);
			}
		} else {
			for (int i=0; i<P*K; i++) {
				solution[i] = solution[i]/scaleDic;
			}
		}
		for (int i=0; i<N*K; i++) {
			solution[P*K+i] = exp(solution[P*K+i]);
		}

		// Normalize weights
		for (int id=0; id<P; id++) {
			double s = 0;
			for (int i=0; i<K; i++) {
				s += solution[id*K+i];
			}
			for (int i=0; i<K; i++) {
				solution[id*K+i] /= s;
			}
		}

		// Normalize each histogram
		for (int i=0; i<K; i++) {
			double s = 0;
			for (int j=0; j<N; j++) {
				s += solution[K*P+i*N+j];
			}
			for (int j=0; j<N; j++) {
				solution[K*P+i*N+j] /= s;
			}
		}

		// Save final solution
		if(exportAtoms)
		{
			// The atoms saved here are the same as those saved at the last iteration
			std::vector<double> base_result(N);
			for (int i=0; i<K; i++) {
				for (int j=0; j<N; j++) {
					base_result[j] = solution[P*K+i*N+j];
				}
				std::ostringstream ss;
				ss << "finalBase_" << std::setfill('0') << std::setw(3) << i;
				std::string filename(ss.str());
				exporter.exportHistogram(base_result,filename);
			}
		}
		if(exportFittings)
		{
			// Trick to compute the final fittings with the same process as the ones during lbfgs :
			// Call evaluate_both one more time : files written will be numbered param.max_iterations
			// If that iteration number is one for which we should export, leave the current one and
			// just make a copy of it for the final files.
			// Otherwise, rename the file so that there isn't an export for that iteration.
			// The finalFitting images will be the same as the one indexed param.max_iterations-1,
			// because it starts at zero
			std::vector<double> gradient(K*P+N*K); // dummy
			// Force exporting
			int M = exportEveryMIter;
			bool final = exportOnlyFinalSolution;
			exportOnlyFinalSolution = false;
			exportEveryMIter = 1;
			// Compute barycenter (and gradients but not used)
			evaluate_both(this,&solution2[0],&gradient[0],K*P+N*K,0);
			// Revert changes made to force exporting
			exportOnlyFinalSolution = final;
			exportEveryMIter = M;
			std::string dir = exporter.mOutputFolderPath + "/";
			std::ostringstream ss;
			ss << "i-fitting_" << std::setfill('0') << std::setw(3) << iteration;
			std::string oldname(ss.str());
			std::vector<std::string> files = getFileListByPattern(dir + oldname + "*");
			for (int i=0; i<files.size(); i++) {
				std::string a = files[i];
				ss.str(""); // Clear the stream
				ss << "finalFitting_" << std::setfill('0') << std::setw(3) << i;
				std::string newname(ss.str());
				// length()+4 because there is the "_XXX" that is the id number of the fitting
				a.replace(a.find("i-fitting"),oldname.length()+4,newname.c_str());

				if(iteration % exportEveryMIter == 0 && !exportOnlyFinalSolution) {
					// Copy the file
					std::ifstream src(files[i].c_str(), std::ios::binary);
					std::ofstream dst(a.c_str(), std::ios::binary);
					dst << src.rdbuf();
				}
				else {
					std::rename(files[i].c_str(), a.c_str());
				}
			}
		}
	}


	// geometric mean used for Bregman projections
	void geomMean(const double* convolved_a, const double* b, double* lambdas, double *result) const {

		memset(&result[0], 0, N*sizeof(result[0]));

		double sumLambdas = 0;
		for (int i=0; i<K; i++) {
			sumLambdas += lambdas[i];
		}
		for (int i=0; i<K; i++) {
			double l = lambdas[i];
			for (int j=0; j<N; j++) {
				result[j] += safe_log(convolved_a[i*N+j])*l;
			}
		}

		for (int j=0; j<N; j++)
			result[j] = exp(result[j]/*/sumLambdas*/);
	}

	void compute_gradient_both(double* resultLa, double* resultDic, double* barycenter, int id) {
		//const int n_iter_gradients = std::max(std::min(3, bary_computation->Niters), std::min(iteration*1, bary_computation->Niters));
		const int n_iter_gradients =  bary_computation->Niters;

		std::vector<double, aligned_allocator<double>  > a(K*N), u(K*N), convu(K*N);
		//std::vector<double, aligned_allocator<double>  > barycenter(N);

		if(!warmRestart)
		{
			b.resize(K*N*(n_iter_gradients+1));
			std::fill(b.begin(), b.begin()+K*N, 1.0);
		}

		conv_b.resize(K*N*n_iter_gradients);
		phi.resize(K*N*(n_iter_gradients+1));
		g.resize(N);
		r.resize(K*N); memset(&r[0], 0, K*N*sizeof(r[0]));

		problem->normalize_values();

		// Bregman Projections
		for (int iter=1; iter<=n_iter_gradients; iter++) {

			//#pragma omp parallel for firstprivate(a)
			problem->kernel.convolveAdjoint(&b[(iter-1)*K*N], &conv_b[(size_t)((iter-1)*K*N)], K);
			for (int i=0; i<K; i++) {
				size_t offset = (size_t)((iter-1)*K*N + i*N);
				for (int j=0; j<N; j++) {
					a[i*N + j] = problem->pdfs[i][0*N + j] / conv_b[offset+j];
				}
			}
			problem->kernel.convolve(&a[0], &phi[iter*K*N], K);

			memset(&barycenter[0], 0, N*sizeof(barycenter[0]));
			for (int i=0; i<K; i++) {
				double l = problem->lambdas[i];
				for (int j=0; j<N; j++) {
					barycenter[j] += safe_log(phi[iter*K*N + i*N+j])*l;
				}
			}
			for (int j=0; j<N; j++)
				barycenter[j] = exp(barycenter[j]);

#pragma omp parallel for
			for (int i=0; i<K; i++) {
				for (int j=0; j<N; j++) {
					b[iter*K*N+i*N+j] = barycenter[j] / phi[iter*K*N + i*N+j];
				}
			}
		}


		std::vector<double, aligned_allocator<double> > n(N), v(K*N,0.0), tmp(K*N), c(K*N), sumv(N);
		loss.gradient(&barycenter[0], &problem->observed_pdf[id*N], N, &g[0]);

		// gradient w.r.t dictionary
		memset(resultDic, 0, K*N*sizeof(resultDic[0]));

		/*if (iteration%2==0)*/ {
			memcpy(&n[0], &g[0], N*sizeof(double));
			for (int sub_iter = n_iter_gradients; sub_iter>=1; sub_iter--) {
				memset(&sumv[0], 0, N*sizeof(sumv[0]));
				for (int i=0; i<K; i++) {
					for (int j=0; j<N; j++) {
						tmp[i*N+j] = (problem->lambdas[i]*n[j]-v[i*N+j]) * b[sub_iter*K*N + i*N+j];
					}
				}
				problem->kernel.convolve(&tmp[0], &c[0], K);
				for (int i=0; i<K; i++) {
					for (int j=0; j<N; j++) {
						resultDic[i*N+j] += c[i*N+j]/conv_b[(sub_iter-1)*K*N+i*N+j];
						tmp[i*N+j] = -problem->pdfs[i][0*N + j]*c[i*N+j]/sqr(conv_b[(sub_iter-1)*K*N+i*N+j]);
					}
				}
				if(sub_iter==1) break;
				problem->kernel.convolveAdjoint(&tmp[0], &v[0], K);
				for (int i=0; i<K; i++) {
					for (int j=0; j<N; j++) {
						v[i*N+j] /= phi[(sub_iter-1)*K*N+i*N+j];

						sumv[j]+=v[i*N+j];
					}
				}
				memcpy(&n[0], &sumv[0], N*sizeof(double));
			}
		}


		///  gradient w.r.t weights


		for (int i=0; i<N; i++) {
			g[i] *= barycenter[i];
		}

		memset(resultLa, 0, K*sizeof(resultLa[0]));

		/*if (iteration%2==1)*/ {
			// applies B recursively
			for (int sub_iter = n_iter_gradients; sub_iter>=1; sub_iter--) {

				//#pragma omp parallel for firstprivate(u, convu)
				for (int i=0; i<K; i++) {
					double dotp = 0;
					for (int j=0; j<N; j++) {
						dotp+= log(phi[sub_iter*K*N + i*N+j]) * g[j];
						u[i*N+j] = (problem->lambdas[i]*g[j] - r[i*N+j])/phi[sub_iter*K*N + i*N+j];
					}
					resultLa[i] += dotp;
				}
				if (sub_iter!=1) {
					problem->kernel.convolve(&u[0], &convu[0], K);
					for (int i=0; i<K; i++) {
						for (int j=0; j<N; j++) {
							convu[i*N+j] *= -problem->pdfs[i][0*N +j] / sqr(conv_b[(sub_iter-1)*K*N + i*N + j]);
						}
					}
					problem->kernel.convolveAdjoint(&convu[0], &r[0], K);
					for (int i=0; i<K*N; i++) {
						r[i] *= b[(sub_iter-1)*K*N + i];
					}

					memset(&g[0], 0, N*sizeof(g[0]));
					for (int i=0; i<K; i++) {
						for (int j=0; j<N; j++)
							g[j] +=r[i*N+j];
					}
				}
			}
		}
	}



void compute_gradient_both_log_signArray(double* resultLa, double* resultDic, double* barycenter, int id) {

	const int n_iter_gradients =  bary_computation->Niters;

	std::vector<double, aligned_allocator<double>  > a(K*N), u(K*N), convu(K*N);

	if(!warmRestart)
	{
		b.resize(K*N*(n_iter_gradients+1));
		std::fill(b.begin(), b.begin()+K*N, 0.0);
	}

	conv_b.resize(K*N*n_iter_gradients);
	phi.resize(K*N*(n_iter_gradients+1));
	g.resize(N);
	r.resize(K*N); memset(&r[0], 0, K*N*sizeof(r[0]));

	problem->normalize_values();

	// Forward log, Backward log + sign array

	for (int iter=1; iter<=n_iter_gradients; iter++) {

		//#pragma omp parallel for firstprivate(a)
		problem->kernel.log_convolveAdjoint(&b[(iter-1)*K*N], &conv_b[(size_t)((iter-1)*K*N)], K);
		for (int i=0; i<K; i++) {
			size_t offset = (size_t)((iter-1)*K*N + i*N);
			for (int j=0; j<N; j++) {
				a[i*N + j] = log(problem->pdfs[i][0*N + j]) - conv_b[offset+j];
			}
		}
		problem->kernel.log_convolve(&a[0], &phi[iter*K*N], K);

		memset(&barycenter[0], 0, N*sizeof(barycenter[0]));
		for (int i=0; i<K; i++) {
			double l = problem->lambdas[i];
			for (int j=0; j<N; j++) {
				barycenter[j] += phi[iter*K*N + i*N+j]*l;
			}
		}

#pragma omp parallel for
		for (int i=0; i<K; i++) {
			for (int j=0; j<N; j++) {
				b[iter*K*N+i*N+j] = barycenter[j] - phi[iter*K*N + i*N+j];
			}
		}
	}

	for (int j=0; j<N; j++)
		barycenter[j] = exp(barycenter[j]);

	std::vector<double, aligned_allocator<double> > n(N), v(K*N,0.0), tmp(K*N), c(K*N), sumv(N);

	loss.gradient(&barycenter[0], &problem->observed_pdf[id*N], N, &g[0]);

	// gradient w.r.t dictionary

	memset(resultDic, 0, K*N*sizeof(resultDic[0]));
	memcpy(&n[0], &g[0], N*sizeof(double));

	unsigned char * signArray = new unsigned char[(K*N+7)/8];


	for (int sub_iter = n_iter_gradients; sub_iter>=1; sub_iter--) {
		memset(&sumv[0], 0, N*sizeof(sumv[0]));
		for (int i=0; i<K; i++) {
			for (int j=0; j<N; j++) {
				tmp[i*N+j] = myAbsLog(problem->lambdas[i]*n[j] + v[i*N+j], signArray, i*N+j) + b[sub_iter*K*N + i*N+j];
			}
		}
		problem->kernel.log_convolve_signArray(&tmp[0], signArray, &c[0], K);
		for (int i=0; i<K; i++) {
			for (int j=0; j<N; j++) {
				resultDic[i*N+j] += myAbsExp(c[i*N+j] - conv_b[(sub_iter-1)*K*N+i*N+j], signArray, i*N+j);

				// We multiply only by positive values, so the sign array stays the same
				tmp[i*N+j] = log(problem->pdfs[i][0*N + j]) + c[i*N+j] - 2.0*(conv_b[(sub_iter-1)*K*N+i*N+j]);
			}
		}
		if(sub_iter==1) break;
		problem->kernel.log_convolve_signArrayAdjoint(&tmp[0], signArray, &v[0], K);
		for (int i=0; i<K; i++) {
			for (int j=0; j<N; j++) {
				// We compute -v
				v[i*N+j] = myAbsExp(v[i*N+j] - phi[(sub_iter-1)*K*N+i*N+j], signArray, i*N+j);
				sumv[j] -= v[i*N+j];
			}
		}
		memcpy(&n[0], &sumv[0], N*sizeof(double));
	}


	//  gradient w.r.t weights

	for (int i=0; i<N; i++) {
		g[i] *= barycenter[i];
	}

	memset(resultLa, 0, K*sizeof(resultLa[0]));

	// applies B recursively
	for (int sub_iter = n_iter_gradients; sub_iter>=1; sub_iter--) {

		//#pragma omp parallel for firstprivate(u, convu)
		for (int i=0; i<K; i++) {
			double dotp = 0;
			for (int j=0; j<N; j++) {
				dotp+= phi[sub_iter*K*N + i*N+j] * g[j];
				u[i*N+j] = myAbsLog(problem->lambdas[i]*g[j] + r[i*N+j], signArray, i*N+j) - phi[sub_iter*K*N + i*N+j];
			}
			resultLa[i] += dotp;
		}

		if (sub_iter!=1) {
			problem->kernel.log_convolve_signArray(&u[0], signArray, &convu[0], K);
			for (int i=0; i<K; i++) {
				for (int j=0; j<N; j++) {
					// We multiply only by positive values, so the sign array stays the same
					convu[i*N+j] += log(problem->pdfs[i][0*N +j]) - 2.0*(conv_b[(sub_iter-1)*K*N + i*N + j]);
				}
			}
			problem->kernel.log_convolve_signArrayAdjoint(&convu[0], signArray, &r[0], K);
			for (int i=0; i<K*N; i++) {
				r[i] = myAbsExp(r[i] + b[(sub_iter-1)*K*N + i], signArray, i);	// We compute -r
			}

			memset(&g[0], 0, N*sizeof(g[0]));
			for (int i=0; i<K; i++) {
				for (int j=0; j<N; j++) {
					g[j] -= r[i*N+j];
				}
			}
		}
	}
	delete[] signArray;
}

// This function doesn't support warmRestart
void compute_gradient_both_log_complex(double* resultLa, double* resultDic, double* barycenter, int id) {

	const int n_iter_gradients =  bary_computation->Niters;

	std::vector<ComplexDataType, aligned_allocator<ComplexDataType> > c_b, c_conv_b, c_phi, c_g, c_r, c_bary;
//	std::vector<double, aligned_allocator<double> > c_b, c_conv_b, c_phi, c_g, c_r;

	std::vector<ComplexDataType, aligned_allocator<ComplexDataType>  > a(K*N), u(K*N), convu(K*N);
//	std::fill(b.begin(), b.begin()+K*N, 0.0);

	g.resize(N);
	c_b.resize(K*N*(bary_computation->Niters+1));
	std::fill(c_b.begin(), c_b.begin()+K*N, 0.0);
	c_conv_b.resize(K*N*bary_computation->Niters);
	c_phi.resize(K*N*(bary_computation->Niters+1));
	c_g.resize(N);
	c_r.resize(K*N); memset(&c_r[0], 0, K*N*sizeof(c_r[0]));
	c_bary.resize(N);
	float logepsilon = (float)log(EPSILON);

	problem->normalize_values();

	// Bregman Projections
	for (int iter=1; iter<=n_iter_gradients; iter++) {

		//#pragma omp parallel for firstprivate(a)
		problem->kernel.log_convolveAdjoint(&c_b[(iter-1)*K*N], &c_conv_b[(size_t)((iter-1)*K*N)], K);
		for (int i=0; i<K; i++) {
			size_t offset = (size_t)((iter-1)*K*N + i*N);
			for (int j=0; j<N; j++) {
				a[i*N + j] = std::log((float)problem->pdfs[i][0*N + j]) - c_conv_b[offset+j];
//				a[i*N + j] = problem->pdfs[i][0*N + j] / conv_b[offset+j];
			}
		}
		problem->kernel.log_convolve(&a[0], &c_phi[iter*K*N], K);
//		problem->kernel.convolve(&a[0], &phi[iter*K*N], K);

		memset(&c_bary[0], 0, N*sizeof(c_bary[0]));
		for (int i=0; i<K; i++) {
			ComplexDataType l = problem->lambdas[i];
			for (int j=0; j<N; j++) {
				c_bary[j] += std::max(logepsilon,std::real(c_phi[iter*K*N + i*N+j]))*l;
//				barycenter[j] += safe_log(phi[iter*K*N + i*N+j])*l;
			}
		}

		// Normal
//		for (int j=0; j<N; j++)
//			barycenter[j] = exp(barycenter[j]);

#pragma omp parallel for
		for (int i=0; i<K; i++) {
			for (int j=0; j<N; j++) {
				c_b[iter*K*N+i*N+j] = c_bary[j] - c_phi[iter*K*N + i*N+j];
//				b[iter*K*N+i*N+j] = barycenter[j] / phi[iter*K*N + i*N+j];
			}
		}
	}

	// Log
	for (int j=0; j<N; j++)
		barycenter[j] = std::real(std::exp(c_bary[j]));


	std::vector<ComplexDataType, aligned_allocator<ComplexDataType> > n(N), v(K*N,0.0), tmp(K*N), c(K*N), sumv(N);
//	std::vector<float, aligned_allocator<float> > n(N);

	loss.gradient(&barycenter[0], &problem->observed_pdf[id*N], N, &g[0]);

	for (int j=0; j<N; j++) c_g[j] = (ComplexDataType) g[j];

	// gradient w.r.t dictionary

	memset(resultDic, 0, K*N*sizeof(resultDic[0]));
	memcpy(&n[0], &c_g[0], N*sizeof(ComplexDataType));

	// Forward log, Backward log
	// Forward normal, Backward normal

	for (int sub_iter = n_iter_gradients; sub_iter>=1; sub_iter--) {
		memset(&sumv[0], 0, N*sizeof(sumv[0]));
		for (int i=0; i<K; i++) {
			for (int j=0; j<N; j++) {
				tmp[i*N+j] = std::log((float)problem->lambdas[i]*n[j] - v[i*N+j]) + c_b[sub_iter*K*N + i*N+j];
//				tmp[i*N+j] = (problem->lambdas[i]*n[j]-v[i*N+j]) * b[sub_iter*K*N + i*N+j];
			}
		}

		problem->kernel.log_convolve(&tmp[0], &c[0], K);
//		problem->kernel.convolve(&tmp[0], &c[0], K);
		for (int i=0; i<K; i++) {
			for (int j=0; j<N; j++) {
				resultDic[i*N+j] += std::real(std::exp(c[i*N+j] - c_conv_b[(sub_iter-1)*K*N+i*N+j]));
//				resultDic[i*N+j] += c[i*N+j]/conv_b[(sub_iter-1)*K*N+i*N+j];
				tmp[i*N+j] = std::log((float)problem->pdfs[i][0*N + j])+c[i*N+j] - 2.0f*(c_conv_b[(sub_iter-1)*K*N+i*N+j]);
//				tmp[i*N+j] = -problem->pdfs[i][0*N + j]*c[i*N+j]/sqr(conv_b[(sub_iter-1)*K*N+i*N+j]);
			}
		}
		if(sub_iter==1) break;
		problem->kernel.log_convolveAdjoint(&tmp[0], &v[0], K);
//		problem->kernel.convolveAdjoint(&tmp[0], &v[0], K);

		ComplexDataType minus = std::log((ComplexDataType)(-1.0f));
		for (int i=0; i<K; i++) {
			for (int j=0; j<N; j++) {
				v[i*N+j] = std::exp(minus - c_phi[(sub_iter-1)*K*N+i*N+j] + v[i*N+j]);
//				v[i*N+j] /= phi[(sub_iter-1)*K*N+i*N+j];
				sumv[j]+= v[i*N+j];
//				sumv[j]+=v[i*N+j];
			}
		}
		memcpy(&n[0], &sumv[0], N*sizeof(ComplexDataType));
	}


	//  gradient w.r.t weights

	for (int i=0; i<N; i++) {
		c_g[i] *= barycenter[i];
	}

	memset(resultLa, 0, K*sizeof(resultLa[0]));

	// applies B recursively
	for (int sub_iter = n_iter_gradients; sub_iter>=1; sub_iter--) {

		//#pragma omp parallel for firstprivate(u, convu)
		for (int i=0; i<K; i++) {
			ComplexDataType dotp = 0;
			for (int j=0; j<N; j++) {
				dotp+= std::max(logepsilon,std::real(c_phi[sub_iter*K*N + i*N+j])) * c_g[j];
//				dotp+= log(phi[sub_iter*K*N + i*N+j]) * g[j];
				u[i*N+j] = std::log((float)problem->lambdas[i]*c_g[j] - c_r[i*N+j]) - c_phi[sub_iter*K*N + i*N+j];
//				u[i*N+j] = (problem->lambdas[i]*g[j] - r[i*N+j])/phi[sub_iter*K*N + i*N+j];
			}
			resultLa[i] += std::real(dotp);
		}

		if (sub_iter!=1) {
			problem->kernel.log_convolve(&u[0], &convu[0], K);
//			problem->kernel.convolve(&u[0], &convu[0], K);
			for (int i=0; i<K; i++) {
				for (int j=0; j<N; j++) {
					convu[i*N+j] += std::log((float)problem->pdfs[i][0*N +j]) - 2.0f*(c_conv_b[(sub_iter-1)*K*N + i*N + j]);
//					convu[i*N+j] *= -problem->pdfs[i][0*N +j] / sqr(conv_b[(sub_iter-1)*K*N + i*N + j]);
				}
			}
			problem->kernel.log_convolveAdjoint(&convu[0], &c_r[0], K);
//			problem->kernel.convolveAdjoint(&convu[0], &r[0], K);

			ComplexDataType minus = std::log((ComplexDataType)(-1.0f));
			for (int i=0; i<K*N; i++) {
				c_r[i] = std::exp(minus + c_r[i] + c_b[(sub_iter-1)*K*N + i]);
//				r[i] *= b[(sub_iter-1)*K*N + i];
			}

			memset(&c_g[0], 0, N*sizeof(c_g[0]));
			for (int i=0; i<K; i++) {
				for (int j=0; j<N; j++)
					c_g[j] += c_r[i*N+j];
//					g[j] +=r[i*N+j];
			}
		}
	}
}


	// for l-bfgs
	// This function evaluates the objective function and its gradient (on both the dictionary and the weights)
	static lbfgsfloatval_t evaluate_both(void *instance, const lbfgsfloatval_t *variables, lbfgsfloatval_t *gradient, const int n, const lbfgsfloatval_t step) {

		WassersteinRegression<KernelType>* regression = (WassersteinRegression<KernelType>*)(instance);
		Problem<KernelType>* prob = regression->problem;
		WassersteinBarycenter<KernelType>* bary = regression->bary_computation;
		int N = prob->N;
		int K = prob->K;
		int P = prob->num_pdfs;
		double scaleDic = regression->scaleDictionary;
		assert(n==P*K+K*N);

		// Transfer the data back to the original domain
		double* variables2 = new double[n];
		memcpy(variables2, variables, n*sizeof(double));
		if (regression->exp_weight) {
			for (int i=0; i<P*K; i++) {
				variables2[i] = exp(variables[i]/scaleDic);
			}
		} else {
			for (int i=0; i<P*K; i++) {
				variables2[i] = variables[i]/scaleDic;
			}
		}
		for (int i=0; i<N*K; i++) {
			variables2[P*K+i] = exp(variables[P*K+i]);
		}
		// Normalize weights
		std::vector<double> s(P, 0);
		for (int i=0; i<P; i++) {
			for (int j=0; j<K; j++) {
				s[i]+= variables2[i*K+j];
			}
			for (int j=0; j<K; j++) {
				variables2[i*K+j] /= s[i];
			}
		}
		// Normalize each histograms
		std::vector<double> s2(K, 0.);
		for (int i=0; i<K; i++) {
			for (int j=0; j<N; j++) {
				s2[i]+=variables2[P*K+i*N+j];
			}
			for (int j=0; j<N; j++) {
				variables2[P*K+i*N+j] /= s2[i];
			}
		}

		// Store the dictionary atoms
		for (int i=0; i<K; i++) {
			for (int j=0; j<N; j++) {
				prob->pdfs[i][j] = variables2[P*K+i*N+j];
			}
		}

		std::vector<double> barycenter(N); // To store one barycenter
		std::vector<double> curGradientDic(K*N); // To store the gradient on dictionary
		memset(gradient, 0, (K*P+N*K) * sizeof(gradient[0])); // Set all values of gradient to 0
		lbfgsfloatval_t lossVal = 0;

		// Containers for histogram and filename batches
		std::vector<std::vector<double> > barycentersBatch(P);
		std::vector<std::string> filenamesBatch(P);
		// Check condition for exporting the histograms
		bool exportHists = regression->exportFittings && !regression->exportOnlyFinalSolution && (regression->iteration % regression->exportEveryMIter == 0);

		// For each histogram in the input dataset
		for (int id =0; id<P; id++) {
			prob->set_weights(&variables2[id*K]); // Store weights
			//prob->normalize_values();
			std::cout<<"|"<<std::flush;

			if(regression->warmRestart)
			{
				// Use last scaling for initialization (warm restart)
				std::copy(regression->b_storage.begin()+K*N*id, regression->b_storage.begin()+K*N*(id+1),regression->b.begin());
			}

			// Compute the barycenter and both gradients
#ifndef COMPUTE_BARYCENTER_LOG
			regression->compute_gradient_both(&gradient[id*K], &curGradientDic[0], &barycenter[0], id);
#else
			regression->compute_gradient_both_log_signArray(&gradient[id*K], &curGradientDic[0], &barycenter[0], id);
#endif

			// Save data (barycenter and filenames) for exporting in batch later
			if(exportHists)
			{
				barycentersBatch[id] = barycenter;
				std::ostringstream ss;
				ss << "i-fitting_" << std::setfill('0');
				ss << std::setw(3) << regression->iteration << "_";
				ss << std::setw(3) << id;
				filenamesBatch[id] = ss.str();
			}

			if(regression->warmRestart)
			{
				// Save last scaling b (for warm restart)
				std::copy(regression->b.begin()+K*N*bary->Niters, regression->b.end(),regression->b_temp.begin()+K*N*id);
			}

			// Compute the loss between the computed barycenter and the input histogram
			// Total loss is the sum over all the input histograms
			double currentLoss = regression->loss.loss(&barycenter[0], &prob->observed_pdf[id*N], prob->N);
//			std::cout<<" "<<currentLoss<<" ";
			lossVal += currentLoss;

			// Consequence of the normalization by the sum on the weights
			// The gradient is not direclty the one computed
			double dotP = dot(&variables2[id*K], &gradient[id*K], K);
			for (int i=0; i<K; i++) {
				gradient[id*K+i] = gradient[id*K+i] - dotP;
			}

			// Consequence of the normalization by the sum on the dictionary
			// The gradient is not direclty the one computed
			// This loop computes the dictionary gradient for one histogram
			for (int i=0; i<K; i++) {
				double dotP = dot(&variables2[P*K+i*N], &curGradientDic[i*N], N);
				for (int j=0; j<N; j++) {
					curGradientDic[i*N+j] = curGradientDic[i*N+j] - dotP;
				}
			}
			// Add dictionary gradient to common gradient array
			for (int i=0; i<K*N; i++) {
				gradient[P*K+i] += curGradientDic[i];
			}
		}

		// Save fittings in batch
		if(exportHists)
		{
			regression->exporter.exportHistogramsBatch(barycentersBatch,filenamesBatch);
		}

		std::cout<<std::endl;
		std::cout<<"loss: "<<lossVal<<" \tstep: "<<step<<std::endl;

		// Apply the chain-rule term due to log-domain transfer
		if (regression->exp_weight) {
			for (int i=0; i<n; i++) {
				gradient[i] *= variables2[i];
			}
		} else {
			for (int i=0; i<N*K; i++) {
				gradient[P*K+i] *= variables2[P*K+i];
			}
		}
		// Scale the weight gradients
		for (int i=0; i<P*K; i++)
			gradient[i] /= scaleDic;


		//regression->compute_gradient_numeric(&test[0]);
		std::cout<<"new iter-----------------"<<std::endl;

		delete[] variables2;

		return lossVal;
	}


	static int progress_both(void *instance, const lbfgsfloatval_t *variables, const lbfgsfloatval_t *gradient, const lbfgsfloatval_t fx,
							 const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n, int k, int ls) {
		WassersteinRegression<KernelType>* regression = (WassersteinRegression<KernelType>*)instance;
		regression->iteration++;
		if(regression->warmRestart)
			printf("LBFGS Iteration %d, total iterations %d :\n", k,regression->iteration);
		else
			printf("Iteration %d:\n", k);
		printf("time elapsed: %f (s)\n", regression->chrono.GetDiffMs()*0.001);
		int K = regression->K;
		int P = regression->problem->num_pdfs;
		int N = regression->N;

		double scaleDic = regression->scaleDictionary;

		// Display fitting variables
		std::cout<<"weights + 10 first values of first atom:"<<std::endl;
		if (regression->exp_weight) {
			for (int i=0; i<K*P; i++) {
				printf("%f ", exp(variables[i]/scaleDic)); // Transfer to original domain
			}
		} else {
			for (int i=0; i<K*P; i++) {
				printf("%f ", variables[i]/scaleDic);
			}
		}
		printf("\n");
		for (int i=0; i<10; i++) {
			printf("%f ", exp(variables[K*P+i]));
		}
		std::cout<<"\nValue = "<<fx<<"\t Step taken: "<<step<<std::endl;

		if(std::isnan(fx))
		{
			std::cout<<"A wild NaN appears ! Canceling lbfgs."<<std::endl;
			return LBFGSERR_CANCELED;
		}

		// Export the dictionary atoms
		if(regression->exportAtoms && !(regression->exportOnlyFinalSolution) && (k % regression->exportEveryMIter == 0))
		{
			std::vector<double> base_result(N);
			for (int i=0; i<K; i++) {
				double s = 0;
				for (int j=0; j<N; j++) {
					base_result[j] = exp(variables[P*K+i*N+j]);
					s += base_result[j];
				}
				for (int j=0; j<N; j++) {
					base_result[j] /= s;
				}
				std::ostringstream ss;
				ss << "i-base_" << std::setfill('0');
				ss << std::setw(3) << regression->iteration << "_";
				ss << std::setw(3) << i;
				std::string filename(ss.str());
				regression->exporter.exportHistogram(base_result,filename);
			}
		}
		return 0;
	}

	// Storage for warm restart
	std::vector<double> b_storage;
	std::vector<double> b_temp;
	// Storage for intermediate results
	std::vector<double, aligned_allocator<double> > b, conv_b, phi, g, r;

	WassersteinBarycenter<KernelType>* bary_computation;
	Problem<KernelType>* problem;
	size_t K, N;
	std::vector<double> prev_solution;
	gradient_type gradient_method;
	PerfChrono chrono;
	const BaseLoss<KernelType> &loss;
	lbfgs_parameter_t lbfgs_param;
	bool firstCall;
	bool exp_weight; // Holds whether the weights are in log-domain
	int iteration;
	int exportEveryMIter;
	// For histogram exportation
	const ExportHistogramBase &exporter;
	// For dual regression (regress_both)
	double scaleDictionary;
	bool exportAtoms;
	bool exportFittings;
	bool exportOnlyFinalSolution; // works for atoms and fittings
	// For warm restart
	bool warmRestart;
	int wrTotalIteration;
};
