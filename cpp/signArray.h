#pragma once

#include <iostream>
#include <bitset>
#include <cmath>


template<typename T>
void setSign(unsigned char *tab, const unsigned int& i, const T& val)
{
    if(val >= 0)
#pragma omp atomic
		tab[i/8] |= (0x01 << (i%8));
	else
#pragma omp atomic
		tab[i/8] &= ~(0x01 << (i%8));
}

float getSign(unsigned char *tab, const unsigned int& i)
{
	return (((tab[i / 8] >> (i % 8)) & 1)<<1) - 1;
}

template<typename T>
T myAbsLog(T val, unsigned char *tab, const unsigned int &i)
{
    setSign(tab,i,val);
	return log(fabs(val));
}

template<typename T>
T myAbsExp(T log_val, unsigned char *tab, const unsigned int &i)
{
	return getSign(tab,i) * exp(log_val);
}

void displaySignArray(unsigned char *tab, unsigned int N)
{
	for(unsigned int i=0; i<N ; ++i)
	{
		std::cout<<getSign(tab,i)<<"\t"; // BigEndian
//         std::cout<<getSign(tab,8*(i/8) + 7 - i%8 )<<"\t"; // LittleEndian
		if((i+1)%8 == 0) std::cout<<std::endl;
	}
	std::cout<<std::endl;
}

int test_signArray()
{
	unsigned char a[2];
	a[0] = 128+64+32+16;
	std::cout<<std::bitset<8>(a[0])<<std::endl;
	setSign(a,1,2);
	std::cout<<std::bitset<8>(a[0])<<std::endl;
	setSign(a,6,-2);
	std::cout<<std::bitset<8>(a[0])<<std::endl;
	setSign(a,1,-2);
	std::cout<<std::bitset<8>(a[0])<<std::endl;
	setSign(a,0,-2);
	std::cout<<std::bitset<8>(a[0])<<std::endl;

	const int N = 80;
	unsigned char tab[N/8] = {1, 2, 4, 8, 16, 32, 64, 128, 255, 127};
	displaySignArray(tab,N);

	const int N2 = 16;
	double values[N2] = {1, -2, -4, 8, -16, 32, 64, 128,  -255, -127, 23, 56, -84, 19, -26, -154};
	double log_values[N2];
	double new_values[N2];
	unsigned char tab3[(N2+7)/8];
	for(int i=0; i<N2; ++i)
	{
		log_values[i] = myAbsLog(values[i],tab3,i);
		std::cout<<log_values[i]<<" ";
	}
	std::cout<<std::endl;
	displaySignArray(tab3,N2);
	for(int i=0; i<N2; ++i)
	{
		new_values[i] = myAbsExp(log_values[i],tab3,i);
		std::cout<<new_values[i]<<" ";
	}
	std::cout<<std::endl;
	return 0;
}

