#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <time.h>
#define INT_MAX 10000
#include <algorithm>
#include <random>

using namespace std;

int minSeq(vector<int> v, int n){
	int min_ele = INT_MAX;
	for(int i =0;i<n;i++){
		if(v[i] <min_ele){
			min_ele = v[i];
		}
	}
	return min_ele;
}


int minPara(vector<int> v, int n){
	int min_ele = INT_MAX;
	#pragma omp parallel for 
		for(int i =0;i<n;i++){
		#pragma omp critical
			if(v[i] <min_ele){
				min_ele = v[i];
			}
		}
	return min_ele;
}

int maxSeq(vector<int> v, int n){
	int maxi=-10000;
	for(int i =0;i<n;i++){
		if(v[i] >maxi){
			maxi=v[i];
		}
	}
	return maxi;
}


int maxPara(vector<int> v, int n){
	int maxi=-10000;
	#pragma omp parallel for
	for(int i =0;i<n;i++){
		if(v[i] >maxi){
			maxi=v[i];
		}
	}
	return maxi;
}






int sum(vector<int> v, int n){
	int s = 0;
	for(auto i:v){
		s+=i;
	}
	
	return s;
}

int sumPara(vector<int> v, int n){
	int s = 0;
	#pragma omp parallel for
		for(auto i:v){
			s+=i;
		}
		
	return s;
}


int averageSeq(vector<int> v,int n){
	int avg = sum(v,n)/n;
	return avg;
}

int averagePara(vector<int> v , int n){
	int avg = sumPara(v,n)/n;
	return avg;
	
}

int main(){
	int N;
	int m = 1000;
	cout<<"Enter the number of Elements : ";
	cin>> N;
	
	vector<int> v(N);
	for(int i=0;i<N;i++){
		v[i]=rand()%1000;
	}
	for(int i=0;i<N;i++){
		cout<<v[i]<<" ";
	}
	cout<<endl;
	
        
   
        // Minimum
        auto start = chrono::high_resolution_clock::now();  
	cout<<"Minimum element " << minSeq(v,N)<<endl;
        auto end  = chrono::high_resolution_clock::now();  
        chrono::duration<double> dur_mins  = end - start;
        cout<< "time taken by seq minimum " << dur_mins.count() <<endl;
	
    	start = chrono::high_resolution_clock::now();
	cout<<"Minimum element (parallel) " << minPara(v,N)<<endl;
	end  = chrono::high_resolution_clock::now();
        chrono::duration<double> dur_minp  = end - start;
        cout<< "time taken by parallel minimum " << dur_minp.count() <<endl;
        
        cout<< "speedup for min : " <<  dur_mins.count() / dur_minp.count() <<endl;
	
	cout<<endl;
	
	// Maximumn
	start = chrono::high_resolution_clock::now(); 
	cout<<"Max element " << maxSeq(v,N)<<endl;
	end  = chrono::high_resolution_clock::now();
        chrono::duration<double> dur_maxs  = end - start;
        cout<< "time taken by seq Max " << dur_maxs.count() <<endl;
        
	start = chrono::high_resolution_clock::now(); 
	cout<<"Max element " << maxPara(v,N)<<endl;
	end  = chrono::high_resolution_clock::now();
        chrono::duration<double> dur_maxp  = end - start;
        cout<< "time taken by parallel Max " << dur_maxp.count() <<endl;
        
        cout<< "speedup for max : " <<  dur_maxs.count() / dur_maxp.count() <<endl;
        
	cout<<endl;
	// Sum
	start = chrono::high_resolution_clock::now(); 
	cout<<"Sum of element " << sum(v,N)<<endl;
	end  = chrono::high_resolution_clock::now();
        chrono::duration<double> dur_sums  = end - start;
        cout<< "time taken by seq sum " << dur_sums.count() <<endl;
        
	start = chrono::high_resolution_clock::now(); 
	cout<<"Sum of element " << sumPara(v,N)<<endl;
	end  = chrono::high_resolution_clock::now();
        chrono::duration<double> dur_sump  = end - start;
        cout<< "time taken by parallel sum " << dur_sump.count() <<endl;
        
        cout<< "speedup for sum : " <<  dur_sums.count() / dur_sump.count() <<endl;
	
	cout<<endl;
	// Average 
	start = chrono::high_resolution_clock::now(); 
	cout<<"Average of element " << averageSeq(v,N)<<endl;
	end  = chrono::high_resolution_clock::now();
        chrono::duration<double> dur_avgs  = end - start;
        cout<< "time taken by seq Average " << dur_avgs.count() <<endl;
        
	start = chrono::high_resolution_clock::now(); 
	cout<<"Average of element " << sumPara(v,N)<<endl;
	end  = chrono::high_resolution_clock::now();
        chrono::duration<double> dur_avgp  = end - start;
        cout<< "time taken by parallel Average " << dur_avgp.count() <<endl;
        
        cout<< "speedup for average : " <<  dur_avgs.count() / dur_avgp.count() <<endl;
	
	
	
	
	return 0;
}
