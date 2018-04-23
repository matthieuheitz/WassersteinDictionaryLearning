#pragma once

template<typename T> T safe_log(const T x) {
	return log(std::max(EPSILON, x));
}

// Parent class for all losses
template<class Kernel>
class BaseLoss {
public:
	BaseLoss() {}
	BaseLoss(Kernel &ker) {}

	virtual double loss(const double* v1, const double* v2, int N) const = 0;
	virtual void gradient(const double* v1, const double* v2, int N, double* res) const = 0;
};


template<class Kernel>
class QuadraticLoss : public BaseLoss<Kernel> {
public:
	QuadraticLoss() {}
	QuadraticLoss(Kernel &ker) {}

	double loss(const double* v1, const double* v2, int N) const {
		double r = 0;
		for (int i=0; i<N; i++) {
			double d = v1[i]-v2[i];
			r += d*d;
		}
		return r;
	}

	void gradient(const double* v1, const double* v2, int N, double* res) const {

		for (int i=0; i<N; i++) {
			res[i] = 2.*(v1[i] - v2[i]);
		}

	}
};

template<class Kernel>
class TVLoss : public BaseLoss<Kernel> {
public:
	TVLoss() {}
	TVLoss(Kernel &ker) {}

	double loss(const double* v1, const double* v2, int N) const {
		double r = 0;
		for (int i=0; i<N; i++) {
			double d = v1[i]-v2[i];
      r += std::abs(d);
		}
		return 0.5*r;
	}

	void gradient(const double* v1, const double* v2, int N, double* res) const {

		for (int i=0; i<N; i++) {
			double d = v1[i]-v2[i];
			res[i] = (d>0?0.5:-0.5);
		}

	}
};

template<class Kernel>
class KLLoss : public BaseLoss<Kernel> {
public:
	KLLoss() {}
	KLLoss(Kernel &ker) {}

	double loss(const double* v1, const double* v2, int N) const {
		double r = 0;
		for (int i=0; i<N; i++) {
			double d = v1[i]*safe_log(v1[i]/v2[i]) - v1[i] + v2[i];
			r += d;
		}
		return r;
	}

	void gradient(const double* v1, const double* v2, int N, double* res) const {

		for (int i=0; i<N; i++) {
			res[i] = safe_log(v1[i] / v2[i]);
		}

	}
};

template<class Kernel>
class WassersteinLoss : public BaseLoss<Kernel> {
public:
	WassersteinLoss(Kernel &ker, int n_iter = 50) : kernel(&ker), num_iter(n_iter) {}

	Kernel * kernel;
	int num_iter;

	double loss(const double* v1, const double* v2, int N) const {return myWloss(v1,v2,N,num_iter);}

	double myWloss(const double* v1, const double* v2, int N, int n_iter) const {

		int Niters = n_iter;
		std::vector<double> a(N, 1.), b(N, 1.), convolution(N);

		// Bregman Projections
		for (int iter=0; iter<Niters; iter++) {

			kernel->convolveAdjoint(&b[0], &convolution[0], 1);
			for (int j=0; j<N; j++) {
				a[j] = v1[j] / convolution[j];
			}
			kernel->convolve(&a[0], &convolution[0], 1);
			for (int j=0; j<N; j++) {
				b[j] = v2[j] / convolution[j];
			}
		}

		double l = 0;
		kernel->convolveAdjoint(&b[0], &convolution[0], 1);
		for (int j=0; j<N; j++) {
			//l+= v1[j]*log(a[j]) + v2[j]*log(b[j]);
			l += safe_log(a[j])*v1[j] + safe_log(b[j])*v2[j] -  a[j]*convolution[j];
		}
		return kernel->gamma*l;
	}

	void gradient(const double* v1, const double* v2, int N, double* res) const {myWgradient(v1,v2,N,res,num_iter);}

	void myWgradient(const double* v1, const double* v2, int N, double* res, int n_iter) const {

		int Niters = n_iter;
		std::vector<double> a(N, 1.), b(N, 1.), convolution(N);

		// Bregman Projections
		for (int iter=0; iter<Niters; iter++) {

			kernel->convolveAdjoint(&b[0], &convolution[0]);
			for (int j=0; j<N; j++) {
				a[j] = v1[j] / convolution[j];
			}
			kernel->convolve(&a[0], &convolution[0]);
			for (int j=0; j<N; j++) {
				b[j] = v2[j] / convolution[j];
			}
		}

		for (int i=0; i<N; i++) {
			res[i]= kernel->gamma*safe_log(a[i]);
		}
	}

};
