#include <iostream>
#include <fstream>
#include <cstdlib>


using namespace std;

int main(){

	ofstream out;
	out.open("data.in");

	for(long long int i=0; i<1e10; i++)
		out << rand()%2 << " ";	
	return 0;
}
