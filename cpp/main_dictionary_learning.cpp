// Uncomment for log-domain stabilization
//#define COMPUTE_BARYCENTER_LOG

#include "inverseWasserstein.h"
#include "histogramIO.h"
#include "cimg/CImg.h"
#include "chrono.h"

#include <map>
#include <string>
#include <math.h>


void dict_learning_2Dimages(int argc, const char* argv[]) {

	std::string usage = "parameters are : "
						"-i <inDir> "
						"-o <outDir> "
						"[-k <numOfAtoms>] "
						"[-l <lossType (1:TV,2:Q,3:W,4:KL)>] "
						"[-n <nIterSinkhorn>] "
						"[-s <scaleDictFactor>] "
						"[-g <gamma>] "
						"[-a <alpha>] "
						"[-x <maxOptimIter>] "
						"[-m <exportEveryMIter>] "
						"[--deterministic] "
						"[--imComplement] "
						"[--allowNegWeights] "
						"[--warmRestart>]";

	if (argc < 4 || std::string(argv[1]) != "-i" || std::string(argv[3]) != "-o")
	{
		std::cout << usage <<std::endl;
		exit(0);
	}

	// Default arguments
	srand(time(NULL));
	std::string inFolder;
	std::string outputFolder = "outputFiles";
	double sigma = 1;
	double gamma = 2*sigma*sigma;
	bool exportAtoms = true;
	bool exportFittings = true;
	bool exportOnlyFinalSolution = false;
	bool allowNegWeights = false;
	bool imComplement = false;
	double alpha = 1.0;
	int maxOptimIter = 200;
	int exportEveryMIter = 0;
	bool warmRestart = false;


	int K = 4;
	int Niter = 25;
	double scaleDictionaryFactor = 100.0;

	int myLoss = 2;

	for (int i = 1; i < argc; i=i+2) {
		bool flag = false;
		if (std::string(argv[i]) == "-i") {
			inFolder = argv[i + 1];
		} else if (std::string(argv[i]) == "-o") {
			outputFolder = argv[i + 1];
		} else if (std::string(argv[i]) == "-k") {
			K = std::stoi(argv[i + 1]);
		} else if (std::string(argv[i]) == "-l") {
			myLoss = std::stoi(argv[i + 1]);
		} else if (std::string(argv[i]) == "-n") {
			Niter = std::stoi(argv[i + 1]);
		} else if (std::string(argv[i]) == "-s") {
			scaleDictionaryFactor = std::stof(argv[i + 1]);
		} else if (std::string(argv[i]) == "-g") {
			gamma = std::stof(argv[i + 1]);
		} else if (std::string(argv[i]) == "-a") {
			alpha = std::stof(argv[i + 1]);
		} else if (std::string(argv[i]) == "-x") {
			maxOptimIter = std::stoi(argv[i + 1]);
		} else if (std::string(argv[i]) == "-m") {
			exportEveryMIter = std::stoi(argv[i + 1]);
		} else if (std::string(argv[i]) == "--deterministic") {
			srand(0);
			flag = true;
		} else if (std::string(argv[i]) == "--allowNegWeights") {
			allowNegWeights = true;
			flag = true;
		} else if (std::string(argv[i]) == "--imComplement") {
			imComplement = true;
			flag = true;
		} else if (std::string(argv[i]) == "--warmRestart") {
			warmRestart = true;
			flag = true;
		} else {
			std::cout << "Not enough or invalid arguments." << std::endl << usage << std::endl;
			exit(0);
		}
		if(flag) i--; // Only skip 1 parameter if argument was flag
	}
	if(exportEveryMIter == 0)
	{
		exportOnlyFinalSolution = true;
		exportEveryMIter = 1; // To avoid div by zero later
	}
	const std::string inputFolder = inFolder;
	const std::wstring wfolder(inputFolder.begin(), inputFolder.end());
	std::vector<std::string> files = get_all_files_names_within_folder(wfolder);
	int P = files.size();

	std::vector<double> normScalings;
	std::vector<double> pdf;
	std::vector<double> observedpdf;
	std::vector<std::vector<double> > bases(K);
	int W, H;
	for (int i=0; i<P; i++) {
//		double scaling = load_img_to_pdf(std::string(inputFolder+"/"+files[i]).c_str(), pdf, W, H);
		double scaling = load_img_to_pdf_gamma_correction(std::string(inputFolder+"/"+files[i]).c_str(), pdf, W, H, alpha, imComplement);
		normScalings.push_back(scaling);
		observedpdf.insert(observedpdf.end(), pdf.begin(), pdf.end());

		if (i>=K) continue;
		bases[i].resize(W*H);
		double s=0;
		// Atom initialization
		for (int j=0; j<W*H; j++) {
//			bases[i][j] = 1.+0.2*rand()/(double)(RAND_MAX);
			bases[i][j] = 1.0;
			s+=bases[i][j];
		}
		for (int j=0; j<W*H; j++) {
			bases[i][j] /= s;
		}
	}
	std::cout<<"nb iter gradient: "<<Niter<<std::endl;

	int N = W*H;
	double scaleDictionary = (N/(double)P)/scaleDictionaryFactor;
	// Container of :
	// 1 - P barycentric weight vectors of size K each
	// 2 - K dictionary atoms of size N each
	std::vector<double> variables(P*K+N*K);
	// Fill the weights
	for (int i=0; i<P; i++) {
		double s = 0;
		for (int j=0; j<K; j++) {
			variables[i*K+j] = rand()/(double)(RAND_MAX);
			s+=variables[i*K+j];
		}
		for (int j=0; j<K; j++) {
			variables[i*K+j]/=s;
		}
	}
	// Fill the atoms
	for (int i=0; i<K; i++) {
		for (int j=0; j<N; j++) {
			variables[P*K+i*N+j] = bases[i][j];
		}
	}

	// Display initialization
	std::cout<<"Initialization (just weights+20): "<<std::endl;
	for (int i=0; i<K*P+20; i++) {
		std::cout<<variables[i]<<" ";
	}
	std::cout<<std::endl;

	#if defined(HAS_HALIDE)
		#ifndef COMPUTE_BARYCENTER_LOG
			typedef GaussianHalide2DKernel KernelType;
		#else
			typedef LogSignArrayGaussianHalide2DKernel<double> KernelType;
		#endif
	#else
		#ifndef COMPUTE_BARYCENTER_LOG
			typedef Gaussian2DKernel KernelType;
		#else
			typedef LogSignArrayGaussian2DKernel KernelType;
		#endif
	#endif

	ExportHistogramToPNGWithScaling exporter(W,H,normScalings,outputFolder,alpha,imComplement);
	KernelType mkernel(gamma, W, H);

	// Utility class that stores all the data
	Problem<KernelType> problem(bases, observedpdf, mkernel, &variables[0], P);

	// Choose the loss
	BaseLoss<KernelType> * loss;
	switch(myLoss)
	{
	case 1: loss = new TVLoss<KernelType>(problem.kernel);			break;
	case 2:	loss = new QuadraticLoss<KernelType>(problem.kernel);	break;
	case 3:	loss = new WassersteinLoss<KernelType>(problem.kernel);	break;
	case 4:	loss = new KLLoss<KernelType>(problem.kernel);			break;
	default : std::cerr<<"Loss index not recognized. Must be in {1,..,4}"<<std::endl; exit(0);
	}

	WassersteinRegression<KernelType> regression(&problem, Niter, GRADIENT_SINKHORN, *loss, exporter,
												scaleDictionary, exportAtoms, exportFittings, exportOnlyFinalSolution, warmRestart);
	regression.exportEveryMIter = exportEveryMIter;
	regression.exp_weight = !allowNegWeights;
	regression.wrTotalIteration = maxOptimIter;

	// Start regression
	regression.regress_both(&variables[0]);


	// Display final solution
	std::cout<<"solution (just weights+20): "<<std::endl;
	for (int i=0; i<K*P+20; i++) {
		std::cout<<variables[i]<<" ";
	}
	std::cout<<std::endl;
}


int main(int argc, const char* argv[]) {

	PerfChrono chrono;
	chrono.Start();

	dict_learning_2Dimages(argc, argv);

	std::cout<<"time taken (s) :"<< chrono.GetDiffMs()*0.001<<std::endl;
	return 0;
}
