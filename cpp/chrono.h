#pragma once

#ifdef _MSC_VER
#include <windows.h>

class PerfChrono
{
    __int64 freq, t0; 

public:
    PerfChrono() {
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    }

    void Start() {
        QueryPerformanceCounter((LARGE_INTEGER*)&t0);
    }

    DWORD GetDiffMs(){
        __int64 t1;
        QueryPerformanceCounter((LARGE_INTEGER*)&t1); 
        return (DWORD)(((t1 - t0) * 1000) / freq); 
    }

    DWORD GetDiffUs() { //micro sec
        __int64 t1; 
        QueryPerformanceCounter((LARGE_INTEGER*)&t1); 
        return (DWORD)(((t1 - t0) * 1000000) / freq); 
    }

    DWORD GetDiffNs(){
        __int64 t1; 
        QueryPerformanceCounter((LARGE_INTEGER*)&t1); 
        return (DWORD)(((t1 - t0) * 1000000000) / freq);
    }

    DWORD GetDiff(UINT unit){
        __int64 t1;
        QueryPerformanceCounter((LARGE_INTEGER*)&t1);
        return (DWORD)(((t1 - t0) * unit) / freq);
    }

    DWORD GetFreq(){
        return (DWORD)freq;
    }
};

#else

#include <sys/time.h>

class PerfChrono
{

public:
	double GetDiffMs(){return GetDiffAbsoluteMs();}

	double GetDiffAbsoluteMs(){
		gettimeofday(&t2, NULL);
		double elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
		elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
		return elapsedTime;
	}
	double GetDiffRelativeMs(){
		gettimeofday(&t2, NULL);
		double elapsedTime = (t2.tv_sec - t3.tv_sec) * 1000.0;      // sec to ms
		elapsedTime += (t2.tv_usec - t3.tv_usec) / 1000.0;   // us to ms
		t3 = t2;
		return elapsedTime;
	}

	void Start() {
		gettimeofday(&t1, NULL);
		t3 = t1;
	}

	struct timeval t1, t2, t3;
};

#endif
