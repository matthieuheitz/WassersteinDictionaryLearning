// File : histogramIO.h
//
// Description :
// Functions and structures to manage the input/output of n-dimensional histograms
// Histograms are also called pdfs for Probability Density Functions
//

// Functions specific to the dimension of the histogram.

#pragma once
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cassert>
#include <algorithm>

#include "cimg/CImg.h"

#if defined(_WIN32)           // Windows

#elif defined(__linux__) || defined(__APPLE__)    // Linux, OS X
	#include <glob.h>
#endif


double load_img_to_pdf(const char* filename, std::vector<double> &result, int &W, int &H);
double load_2D_img_to_pdf(const char* filename, std::vector<double> &result, int &W, int &H);
double load_rgb_img_to_pdf_grayscale_luma(const char* filename, std::vector<double> &result, int &W, int &H);
double load_img_to_pdf_gamma_correction(const char* filename, std::vector<double> &result, int &W, int &H, double alpha, bool complement);
double load_csv_to_pdf(const char* filename, std::vector<double> &result);
void save_pdf(const char* filename, int W, int H, const std::vector<double> &val, double scaling);
void save_pdf_gamma_correction(const char* filename, int W, int H, const std::vector<double> &val, double alpha, bool im_complement, double scaling);

// Helper functions
double lerp(double v0, double v1, double t);
std::vector<double> quantile(const std::vector<double>& inData, const std::vector<double>& probs);
std::vector<std::string> getFileListByPattern(const std::string& pat);
std::vector<std::string> get_all_files_names_within_folder(std::wstring folder);

// Abstract export class
class ExportHistogramBase{
public:
	ExportHistogramBase(std::string outputFolderPath) : mOutputFolderPath(outputFolderPath) {}

	// Must return 0 if export is successful, other values otherwise
	virtual int exportHistogram(const std::vector<double> &results, std::string filenameNoExtension) const = 0 ;

	// Must return 0 if export is successful, other values otherwise
	virtual int exportHistogramsBatch(const std::vector<std::vector<double>> &results, std::vector<std::string> filenamesNoExtension) const = 0 ;

//protected:
	std::string mOutputFolderPath;
};


// Export a nD histogram to a CSV file
class ExportHistogramToCSV : public ExportHistogramBase {

public:
	ExportHistogramToCSV(std::string outputFolderPath = "outputFiles") : ExportHistogramBase(outputFolderPath)
	{}

	int exportHistogram(const std::vector<double> &results, std::string filenameNoExtension) const
	{
		std::ofstream ofs;
		std::string filename = mOutputFolderPath + "/" + filenameNoExtension + ".csv";
		ofs.open(filename);
		if(!ofs.is_open())
		{
			std::cerr<<"Unable to write file "<<filename<<std::endl;
			return -1;
		}
		for(int i=0; i<results.size(); ++i)
		{
			ofs<<results[i]<<"\n";
		}
		ofs.close();
		return 0;
	}

	int exportHistogramsBatch(const std::vector<std::vector<double>> &results, std::vector<std::string> filenamesNoExtension) const
	{
		if(results.size() != filenamesNoExtension.size())
		{
			std::cerr<<"Histogram vector and filename vector are of different sizes"<<std::endl;
			return -1;
		}
		int ret;
		bool success = true;
		for(int i=0; i<results.size(); i++)
		{
			ret = this->exportHistogram(results[i],filenamesNoExtension[i]);
			if(ret != 0) success = false;
		}
		return (success ? 0 : -1);
	}
};

// Export a 2D histogram to a graylevel image
class ExportHistogramToPNG : public ExportHistogramBase {

public:
	int mWidth, mHeight;

	ExportHistogramToPNG(int width, int height, std::string outputFolderPath = "outputFiles") :
		mWidth(width), mHeight(height), ExportHistogramBase(outputFolderPath) {}

	int exportHistogram(const std::vector<double> &results, std::string filenameNoExtension) const
	{
		std::string filename = mOutputFolderPath + "/" + filenameNoExtension + ".png";
		assert(results.size() == mWidth*mHeight);

		std::vector<double> probs = {0.95};
		std::vector<double> quantiles = quantile(results,probs);
		save_pdf(filename.c_str(), mWidth, mHeight, results, 255.0*probs[0]/quantiles[0]);
		return 0;
	}

	int exportHistogramsBatch(const std::vector<std::vector<double>> &results, std::vector<std::string> filenamesNoExtension) const
	{
		if(results.size() != filenamesNoExtension.size())
		{
			std::cerr<<"Histogram vector and filename vector are of different sizes"<<std::endl;
			return -1;
		}
		int ret;
		bool success = true;
		for(int i=0; i<results.size(); i++)
		{
			ret = this->exportHistogram(results[i],filenamesNoExtension[i]);
			if(ret != 0) success = false;
		}
		return (success ? 0 : -1);
	}
};

// Export a 2D histogram to a graylevel image
// Holds a vector of scaling values to apply to histogram when exporting them.
class ExportHistogramToPNGWithScaling : public ExportHistogramBase {

public:
	int mWidth, mHeight;
	std::vector<double> mScalings;
	float mAlpha; // Value for the gamma-correction
	bool mComplement; // Take the image complement

	ExportHistogramToPNGWithScaling(int width, int height, std::vector<double>& scalings, std::string outputFolderPath = "outputFiles", float alpha = 1, bool imComplement = false) :
		mWidth(width), mHeight(height), mScalings(scalings), mAlpha(alpha), mComplement(imComplement), ExportHistogramBase(outputFolderPath) {}

	int exportHistogram(const std::vector<double> &results, std::string filenameNoExtension) const
	{
		std::string filename = mOutputFolderPath + "/" + filenameNoExtension + ".png";
		assert(results.size() == mWidth*mHeight);

		std::vector<double> probs = {0.95};
		std::vector<double> quantiles = quantile(results,probs);
		save_pdf_gamma_correction(filename.c_str(), mWidth, mHeight, results, mAlpha, mComplement, probs[0]/quantiles[0]);
		return 0;
	}

	int exportHistogramWithScaling(const std::vector<double> &results, std::string filenameNoExtension, int i) const
	{
		std::string filename = mOutputFolderPath + "/" + filenameNoExtension + ".png";
		assert(results.size() == mWidth*mHeight);
		if(i >= mScalings.size())
		{
			std::cerr<<"Index "<<i<<" is bigger than the vector of scalings' size ("<<mScalings.size()<<")"<<std::endl;
			return -1;
		}
		save_pdf_gamma_correction(filename.c_str(), mWidth, mHeight, results, mAlpha, mComplement, mScalings[i]);
		return 0;
	}

	// In this class, this function is a little special.
	// It is made so that:
	// - you load 2D images in a batch, and store the scalings in mScalings.
	// - your algorithm computes reconstructions of each of these images
	// - you save the reconstructions in a batch with this function so that the right scaling can be applied to the right reconstruction.
	int exportHistogramsBatch(const std::vector<std::vector<double>> &results, std::vector<std::string> filenamesNoExtension) const
	{
		if(results.size() != filenamesNoExtension.size())
		{
			std::cerr<<"Histogram vector and filename vector are of different sizes"<<std::endl;
			return -1;
		}
		int ret;
		bool success = true;
		for(int i=0; i<results.size(); i++)
		{
			// Here, the subtlety is in the "i" at the end that says which scaling to use.
			ret = this->exportHistogramWithScaling(results[i],filenamesNoExtension[i],i);
			if(ret != 0) success = false;
		}
		return (success ? 0 : -1);
	}
};


// Export a nD histogram to a raw binary file
class ExportHistogramToRAW : public ExportHistogramBase {

public:
	ExportHistogramToRAW(std::string outputFolderPath = "outputFiles") : ExportHistogramBase(outputFolderPath)
	{}

	int exportHistogram(const std::vector<double> &results, std::string filenameNoExtension) const
	{
		std::ofstream ofs;
		std::string filename = mOutputFolderPath + "/" + filenameNoExtension + ".raw";
		ofs.open(filename, std::ios::out | std::ios::binary);
		if(!ofs.is_open())
		{
			std::cerr<<"Unable to write file "<<filename<<std::endl;
			return -1;
		}
		ofs.write(reinterpret_cast<const char*>(&results[0]), results.size()*sizeof(double));
		ofs.close();
		return 0;
	}

	int exportHistogramsBatch(const std::vector<std::vector<double>> &results, std::vector<std::string> filenamesNoExtension) const
	{
		if(results.size() != filenamesNoExtension.size())
		{
			std::cerr<<"Histogram vector and filename vector are of different sizes"<<std::endl;
			return -1;
		}
		int ret;
		bool success = true;
		for(int i=0; i<results.size(); i++)
		{
			ret = this->exportHistogram(results[i],filenamesNoExtension[i]);
			if(ret != 0) success = false;
		}
		return (success ? 0 : -1);
	}
};




double load_img_to_pdf(const char* filename, std::vector<double> &result, int &W, int &H) {

	cimg_library::CImg<unsigned char> cimg(filename);
	std::cout<<"loading "<<std::string(filename)<<std::endl;
	W = cimg.width();
	H = cimg.height();
	result.resize(W*H);
	double sum = 0;
	for (int i=0; i<W*H; i++) {
		result[i] = cimg.data()[i]+1E-12;
		sum+=result[i];
	}
	for (int i=0; i<W*H; i++) {
		result[i] /= sum;
	}
	return sum;
}

double load_2D_img_to_pdf(const char* filename, std::vector<double> &result, int &W, int &H) {

	std::string f(filename);
	double r = load_img_to_pdf(filename,result,W,H);
	if(W == 0 || H == 0) return -1;
	return r;
}

double load_rgb_img_to_pdf_grayscale_luma(const char* filename, std::vector<double> &result, int &W, int &H) {

	cimg_library::CImg<unsigned char> cimg(filename);
	printf("loading %s\n",filename);
	W = cimg.width();
	H = cimg.height();
	result.resize(W*H);
	double sum = 0;
	for (int i=0; i<W*H; i++) {
		result[i] = 0.216*cimg.data(i,0,0,0)[0] + 0.7152*cimg.data(i,0,0,1)[0] + 0.0722*cimg.data(i,0,0,2)[0] + 1E-12;
		sum+=result[i];
	}
	for (int i=0; i<W*H; i++) {
		result[i] /= sum;
	}
	return sum;
}

double load_img_to_pdf_gamma_correction(const char* filename, std::vector<double> &result, int &W, int &H, double alpha, bool complement) {

	cimg_library::CImg<unsigned char> cimg(filename);
	std::cout<<"loading "<<std::string(filename)<<std::endl;
	W = cimg.width();
	H = cimg.height();
	result.resize(W*H);
	double sum = 0;
	if(complement) {
		for (int i=0; i<W*H; i++) {
			result[i] = std::pow((255.0-cimg.data()[i])/255.0, alpha) + 1E-12;
			sum+=result[i];
		}
	} else {
		for (int i=0; i<W*H; i++) {
			result[i] = std::pow(cimg.data()[i]/255.0, alpha) + 1E-12;
			sum+=result[i];
		}
	}
	for (int i=0; i<W*H; i++) {
		result[i] /= sum;
	}
	return sum;
}


// For 1D histogram with values in one column in a csv file.
double load_csv_to_pdf(const char* filename, std::vector<double> &result) {

	std::ifstream ifs;
	ifs.open(filename);
	if(!ifs.is_open())
	{
		std::cerr<<"Unable to read file "<<filename<<std::endl;
		return -1;
	}
	double v,sum=0;
	while(ifs.good())
	{
		ifs>>v;
		sum +=v;
		result.push_back(v);
	}
	ifs.close();
	for(int i=0; i<result.size(); ++i)
	{
		result[i] /= sum;
	}
	return sum;

}

void save_pdf(const char* filename, int W, int H, const std::vector<double> &val, double scaling) {

	std::vector<unsigned char> deinterleaved(W*H * 3);
	for (int i = 0; i < W*H; i++) {
		deinterleaved[i] = std::min(255., std::max(0., val[i] * scaling));
		deinterleaved[i + W*H] = deinterleaved[i];
		deinterleaved[i + 2 * W*H] = deinterleaved[i];
	}

	cimg_library::CImg<unsigned char> cimg(&deinterleaved[0], W, H, 1, 3);
	cimg.save(filename);

}

void save_pdf_gamma_correction(const char* filename, int W, int H, const std::vector<double> &val, double alpha, bool im_complement, double scaling) {

	std::vector<unsigned char> deinterleaved(W*H * 3);
	if(im_complement) {
		for (int i = 0; i < W*H; i++) {
			deinterleaved[i] = std::min(255., std::max(0., 255. - std::pow(val[i]*scaling, 1.0/alpha) * 255.));
			deinterleaved[i + W*H] = deinterleaved[i];
			deinterleaved[i + 2 * W*H] = deinterleaved[i];
		}
	} else {
		for (int i = 0; i < W*H; i++) {
			deinterleaved[i] = std::min(255., std::max(0., std::pow(val[i]*scaling, 1.0/alpha) * 255.));
			deinterleaved[i + W*H] = deinterleaved[i];
			deinterleaved[i + 2 * W*H] = deinterleaved[i];
		}
	}
	cimg_library::CImg<unsigned char> cimg(&deinterleaved[0], W, H, 1, 3);
	cimg.save(filename);
}

double lerp(double v0, double v1, double t)
{
	return (1 - t)*v0 + t*v1;
}

// Taken from Yury : http://stackoverflow.com/a/37708864/4195725
std::vector<double> quantile(const std::vector<double>& inData, const std::vector<double>& probs)
{
	if (inData.size() <= 2 || probs.empty())
	{
		throw std::runtime_error("Invalid input");
	}

	std::vector<double> data = inData;
	std::sort(data.begin(), data.end());
	std::vector<double> quantiles;

	for (size_t i = 0; i < probs.size(); ++i)
	{
		double center = lerp(-0.5, data.size() - 0.5, probs[i]);

		size_t left = std::max(int64_t(std::floor(center)), int64_t(0));
		size_t right = std::min(int64_t(std::ceil(center)), int64_t(data.size() - 1));

		double dataLeft = data.at(left);
		double dataRight = data.at(right);

		double quantile = lerp(dataLeft, dataRight, center - left);

		quantiles.push_back(quantile);
	}

	return quantiles;
}


#if defined(_WIN32)

static std::vector<std::string> findFiles(const std::string& pat){
	std::string filename;
	std::vector<std::string> listOfFileNames;

	WIN32_FIND_DATAA findFileData;
	HANDLE myHandle = FindFirstFileA(pat.c_str(),&findFileData);

	if( myHandle != INVALID_HANDLE_VALUE)
	{
		do
		{
			filename = findFileData.cFileName;
			listOfFileNames.push_back( filename);
		} while (FindNextFileA(myHandle, &findFileData) != 0);
	}
	FindClose(myHandle);
	std::sort(listOfFileNames.begin(), listOfFileNames.end());
	return listOfFileNames;
}

#elif defined(__linux__) || defined(__APPLE__)

static std::vector<std::string> glob(const std::string& pat){
	glob_t glob_result;
	glob(pat.c_str(),GLOB_TILDE | GLOB_BRACE,NULL,&glob_result);
	std::vector<std::string> ret;
	for(unsigned int i=0;i<glob_result.gl_pathc;++i){
		ret.push_back(std::string(glob_result.gl_pathv[i]));
	}
	globfree(&glob_result);
	return ret;
}

#endif

std::vector<std::string> getFileListByPattern(const std::string& pat)
{
#if defined(_WIN32)
	return findFiles(pat);
#elif defined(__linux__) || defined(__APPLE__)
	return glob(pat);
#endif
}


std::vector<std::string> get_all_files_names_within_folder(std::wstring folder) {
	#if defined(_WIN32)
		std::vector<std::string> names;
		TCHAR search_path[200];
		std::string strpath(folder.begin(), folder.end());
		sprintf(search_path, "%s/*.*", strpath.c_str());
		WIN32_FIND_DATA fd;
		HANDLE hFind = ::FindFirstFile(search_path, &fd);
		if (hFind != INVALID_HANDLE_VALUE) {
			do {
				// read all (real) files in current folder
				// , delete '!' read other 2 default folder . and ..
				if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
					std::string str(fd.cFileName);
					names.push_back(str);
				}
			} while (::FindNextFile(hFind, &fd));
			::FindClose(hFind);
		}
		return names;
	#elif defined(__unix__) || defined(__APPLE__) // Linux and OSX
		std::vector<std::string> names;
		std::string folderPath(folder.begin(), folder.end());
		DIR *dir;
		struct dirent *ent;
		std::cout<<"Reading in folder "<<folderPath.c_str()<<std::endl;
		if ((dir = opendir (folderPath.c_str())) != NULL) {
		  /* print all the files and directories within directory */
		  while ((ent = readdir (dir)) != NULL) {
			if(std::strcmp(ent->d_name,".") != 0 && std::strcmp(ent->d_name,"..") != 0)
			{
				names.push_back(ent->d_name);
			}
		  }
		  closedir (dir);
		} else {
		  /* could not open directory */
		  perror ("");
		  return names;
		}
		std::sort(names.begin(), names.end());
		return names;
	#endif
}
