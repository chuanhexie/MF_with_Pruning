#include<iostream>
#include<iomanip>
#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<fstream>
#include<sstream>
#include<time.h>
using namespace std;
#define feature_size 8
#define datasize 8000
#define STEP 10000
#define STEPR 100
#define ALAPA 0.0002
#define BETA 0.1
#define PMIN 1.0
#define MINP 0.0
#define PMAX 0.6

int Rlen=0,Tlen=0;

int main() {
    ifstream infile;
    infile.open("train_small_2.txt", ios::in);
    ofstream of1("trainset.txt");
    ofstream of2("testset.txt");
    stringstream ss;
    if (!infile.is_open()) cout<<"NO\n";
    char buffer[256];
    int a,b,c;
    int train[datasize][3], test[datasize][3];
    int MAXa=0, MAXb=0, MAXc=0, MINc=-1;
    int seed = time(0);
    srand(seed);
    while(infile.getline(buffer,256) && buffer!=NULL) {
        ss << buffer;
        ss>>a>>b>>c;
        ss.clear();
        if (rand()%10>0) {
            of1<<buffer<<endl;
            Rlen++;
        }
        else {
            of2<<buffer<<endl;
            Tlen++;
        }
        if (a>MAXa) MAXa = a;
        if (b>MAXb) MAXb = b;
        if (c>MAXc) MAXc = c;
        if (c<MINc || MINc==-1) MINc = c;
    }
    infile.close();of1.close();of2.close();
    int N=MAXa+1, M=MAXb+1, K=feature_size;
    cout<<N<<' '<<M<<' '<<K<<"\n";
    cout<<Rlen<<' '<<Tlen<<endl;

    
    system("pause");
    return 0;
}
