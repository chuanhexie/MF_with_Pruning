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
#define ALAPA 0.002
#define BETA 0.1

int Rlen=0,Tlen=0;

double cal_rmse(const double *R,const double *P,const double *Q,const int N,const int M,const int K,const int L) {
    double loss = 0;
    for(int i=0; i<N; ++i) {
        for(int j=0; j<M; ++j) {
            if(R[i*M+j] > 0 ) {
                double error=0;
                for (int k=0; k<K; ++k)
                    error += P[i*K+k]*Q[j*K+k];
                loss += pow(R[i*M+j]-error,2);
            }
        }
    }
    return loss/L;
}

void matrix_factorization(const double *R,double *P,double *Q,int N,int M,int K,int steps=STEP,float alpha=ALAPA,float beta=BETA) {
    double error;
    for(int step =0; step<steps; ++step) {
        for(int i=0; i<N; ++i) {
            for(int j=0; j<M; ++j) {
                if(R[i*M+j]>0) {
                    error = R[i*M+j];
                    for(int k=0; k<K; ++k) error -= P[i*K+k]*Q[j*K+k]; //
                    for(int k=0; k<K; ++k) {
                        P[i*K+k] += alpha * (2 * error * Q[j*K+k] - beta * P[i*K+k]);//
                        Q[j*K+k] += alpha * (2 * error * P[i*K+k] - beta * Q[j*K+k]);
                        /*
                            P(i,k) += alpha * e(i,j) * Q(k,j) - beta * P(i,j)
                            Q(i,k) += alpha * e(i,j) * P(k,j) - beta * Q(i,j)
                        */
                    }
                }
            }
        }

        if(step == STEP-1) {
            cout<<"Final loss:"<<cal_rmse(R,P,Q,N,M,K,Rlen)<<endl;
            break;
        }
        else if (step%(int)(0.1*STEP)==0)
            cout<<"loss:"<<cal_rmse(R,P,Q,N,M,K,Rlen)<<endl;
    }
}

void print_rst(double* T, double* P, double* Q,int N,int M,int K,int Tlen){
    double loss = 0, loss1 = 0, lossabs1 = 0;
    for(int i=0; i<N; ++i) {
        for(int j=0; j<M; ++j) {
            if(T[i*M+j] > 0 ) {
                double error=0;
                for (int k=0; k<K; ++k)
                    error+=P[i*K+k]*Q[j*K+k];
                loss += pow(error-T[i*M+j], 2);
                loss1 += error-T[i*M+j];
                lossabs1 += abs(error-T[i*M+j]);
            }
        }
    }
    cout<<"test-set RMSE: "<<loss/Tlen<<"\n1-toldiff: "<<loss1<<"\n1-absdiff: "<<lossabs1/Tlen<<endl<<endl;
}


int main() {
    ifstream infile1("trainset.txt");
    ifstream infile2("testset.txt");
    if (!infile1.is_open() || !infile2.is_open()) cout<<"NO\n";
    stringstream ss;
    char buffer[256];
    int a,b,c;
    int train[datasize][3], test[datasize][3];
    int MAXa=0, MAXb=0, MAXc=0, MINc=-1;
    int seed = time(0);
    srand(seed);

    while(infile1.getline(buffer,256) && buffer!=NULL) {
        ss << buffer;
        ss>>a>>b>>c;
        ss.clear();
        train[Rlen][0] = a;
        train[Rlen][1] = b;
        train[Rlen][2] = c;
        Rlen++;
        if (a>MAXa) MAXa = a;
        if (b>MAXb) MAXb = b;
        if (c>MAXc) MAXc = c;
        if (c<MINc || MINc==-1) MINc = c;
    }
    while(infile2.getline(buffer,256) && buffer!=NULL) {
        ss << buffer;
        ss>>a>>b>>c;
        ss.clear();
        test[Tlen][0] = a;
        test[Tlen][1] = b;
        test[Tlen][2] = c;
        Tlen++;
        if (a>MAXa) MAXa = a;
        if (b>MAXb) MAXb = b;
        if (c>MAXc) MAXc = c;
        if (c<MINc || MINc==-1) MINc = c;
    }
    infile1.close();
    infile2.close();
    int N=MAXa+1, M=MAXb+1, K=feature_size;
    cout<<N<<' '<<M<<' '<<K<<"\n";
    cout<<Rlen<<' '<<Tlen<<endl;

    double *R = new double[N*M];
    double *T = new double[N*M];
    double *P = new double[N*K];
    double *P2 = new double[N*K];
    double *PO = new double[N*K];
    int *PU = new int[N*K];
    double *Q = new double[M*K];
    double *Q2 = new double[M*K];
    double *QO = new double[M*K];

    /*for(int i=0; i<N; ++i)
    {
        for(int j=0; j<M; ++j)
            cout<< R[i*M+j]<<',';
        cout<<endl;
    }*/
    for(int i=0; i<Rlen; i++) {
        R[train[i][0]*M+train[i][1]] = train[i][2];
    }
    for(int i=0; i<Tlen; i++) {
        T[test[i][0]*M+test[i][1]] = test[i][2];
    }


    for(int i=0; i<N; ++i) //
        for(int j=0; j<K; ++j) {
            P[i*K+j]=1;
            PU[i*K+j]=rand()%10/10.0;
        }
    for(int i=0; i<M; ++i)
        for(int j=0; j<K; ++j)
            Q[i*K+j]=rand()%10/10.0; 

    matrix_factorization(R,P,Q,N,M,K); // 初次进行矩阵分解得到最初的结果
    
    /*cout<<"\n\n\norignal P P P P\n";
    for(int i=0; i<N; ++i) {
        cout<<(i/1000)%10<<(i/100)%10<<(i/10)%10<<i%10<<": ";
        for(int j=0; j<K; ++j)
            cout<<setprecision(4)<<(char)((P[i*K+j]<0)?7:32)<<fixed<<P[i*K+j]<<' ';
        cout<<endl;
    }
    cout<<endl;*/

    cout<<"train-set RMSE: "<<cal_rmse(R,P,Q,N,M,K,Rlen)<<endl;
    print_rst( T, P, Q, N, M, K, Tlen);
    
    delete [] P, P2, PU, PO, Q, Q2, QO, R, T;

    system("pause");
    return 0;
}
