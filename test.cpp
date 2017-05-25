#include<iostream>
#include<iomanip>
#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<fstream>
#include<sstream>
#include<time.h>
using namespace std;
#define feature_size 32
#define datasize 8000
#define STEP 1000
#define STEPR 50
#define ALAPA 0.002
#define BETA 0.1
#define PMIN 0.2
#define MINP 0.0
#define PMAX 0.6

int Rlen=0,Tlen=0;

double cal_rmse(const double *R,const double *B,const double *C,const double *P,const double *Q,int N,int M,int K,int L) {
    double loss = 0;
    for(int i=0; i<N; ++i) {
        for(int j=0; j<M; ++j) {
            if(R[i*M+j] > 0 ) {
                double error = B[i] + C[j];
                for (int k=0; k<K; ++k)
                    error += P[i*K+k] * Q[j*K+k];
                loss += pow(R[i*M+j]-error,2);
            }
        }
    }
    return loss/L;
}

bool cal_user_error(const double *R,const double *B,const double *C,const double *P,const double *Q,int N,int M,int K,int V) {
    double loss = 0;
    for(int j=0; j<M; ++j) {
        if(R[V*M+j] > 0 ) {
            double error = B[V] + C[j];
            for (int k=0; k<K; ++k)
                error += P[V*K+k] * Q[j*K+k];
            if(abs(R[V*M+j]-error) > 0.5) return false;
        }
    }
    return true;
}

bool cal_item_error(const double *R,const double *B,const double *C,const double *P,const double *Q,int N,int M,int K,int V) {
    double loss = 0;
    for(int i=0; i<N; ++i) {
        if(R[i*M+V] > 0 ) {
            double error = B[i] + C[V];
            for (int k=0; k<K; ++k)
                error += P[i*K+k] * Q[V*K+k];
            if(abs(R[i*M+V]-error) > 0.5) return false;
        }
    }
    return true;
}

void matrix_factorization(const double *R,double *B,double *C,double *P,double *Q,int N,int M,int K,int steps=STEP,float alpha=ALAPA,float beta=BETA) {
    double error;
    for(int step =0; step<steps; ++step) {
        for(int i=0; i<N; ++i) {
            for(int j=0; j<M; ++j) {
                if(R[i*M+j]>0) {
                    error = R[i*M+j] - B[i] - C[j];
                    for(int k=0; k<K; ++k) error -= P[i*K+k] * Q[j*K+k];
                    for(int k=0; k<K; ++k) {
                        B[i] += alpha * (2 * error);
                        C[i] += alpha * (2 * error);
                        P[i*K+k] += alpha * (2 * error * Q[j*K+k] - beta * P[i*K+k]);
                        Q[j*K+k] += alpha * (2 * error * P[i*K+k] - beta * Q[j*K+k]);
                    }
                }
            }
        }

        if(step == STEP-1) {
            cout<<"Final loss:"<<cal_rmse(R,B,C,P,Q,N,M,K,Rlen)<<endl;
            break;
        }
        else if (step%(int)(0.1*STEP)==0)
            cout<<"loss:"<<cal_rmse(R,B,C,P,Q,N,M,K,Rlen)<<endl;
    }
}

/*void matrix_recounstruction(const double *R,double *B,double *P,double *Q,int N,int M,int K,int steps=STEPR,float alpha=ALAPA,float beta=BETA) {
    double error;
    for(int step =0; step<steps; ++step) { // 重建与初次的差异步骤不大，但需要判断为0的位置不进行更新
        for(int i=0; i<N; ++i) {
            for(int j=0; j<M; ++j) {
                if(R[i*M+j]>0) {
                    error = R[i*M+j];
                    for(int k=0; k<K; ++k) error -= (P[i*K+k]+B[i])*Q[j*K+k];
                    for(int k=0; k<K; ++k)
                        if(P[i*K+k]!=MINP) {
                            B[i] += alpha * (2 * error * Q[i*K+k]);
                            // B[i] += alpha * (2 * error * Q[j*K+k] - 2 * beta * B[i]);
                            P[i*K+k] += alpha * (2 * error * Q[j*K+k] - beta * P[i*K+k]);
                            Q[j*K+k] += alpha * (2 * error * (P[i*K+k] + B[i]) - beta * Q[j*K+k]);
                        }
                }
            }
        }
    }
}*/

void user_vector_reconstruction(const double *R,double *B,double *C,double *P,double *Q,int N,int M,int K,int V,int steps=STEPR,float alpha=ALAPA,float beta=BETA) {
    double error;
    for(int step =0; step<steps; ++step) { // 重建与初次的差异步骤不大，但需要判断为0的位置不进行更新
        for(int j=0; j<M; ++j) {
            if(R[V*M+j]>0) {
                error = R[V*M+j] - B[V] - C[j];
                for(int k=0; k<K; ++k) error -= P[V*K+k] * Q[j*K+k];
                for(int k=0; k<K; ++k)
                    if(P[V*K+k]!=MINP) {
                        B[V] += alpha * (2 * error * Q[j*K+k] - beta * B[V]);
                        P[V*K+k] += alpha * (2 * error * Q[j*K+k] - beta * P[V*K+k]);
                        /*
                            B(i)   += alpha * e(i,j) * Q(k,j);
                            P(i,j) += alpha * e(i,j) * Q(k,j) - beta * P(i,j)
                        */
                    }
            }
        }
    }
}

void item_vector_reconstruction(const double *R,double *B,double *C,double *P,double *Q,int N,int M,int K,int V,int steps=STEPR,float alpha=ALAPA,float beta=BETA) {
    double error;
    for(int step =0; step<steps; ++step) { // 重建与初次的差异步骤不大，但需要判断为0的位置不进行更新
        for(int i=0; i<N; ++i) {
            if(R[i*M+V]>0) {
                error = R[i*M+V] - B[i] - C[V];
                for(int k=0; k<K; ++k) error -= P[i*K+k] * Q[V*K+k];
                for(int k=0; k<K; ++k)
                    if(Q[V*K+k]!=MINP) {
                        C[V] += alpha * (2 * error * P[i*K+k] - beta * C[V]);
                        Q[V*K+k] += alpha * (2 * error * P[i*K+k] - beta * Q[V*K+k]);
                    }
            }
        }
    }
}

void print_rst(double* T, double* B, double* C, double* P, double* Q,int N,int M,int K,int Tlen) {
    double loss = 0, loss1 = 0, lossabs1 = 0, error, cnt=0;
    for(int i=0; i<N; ++i) {
        for(int j=0; j<M; ++j) {
            if(T[i*M+j] > 0 ) {
                error=B[i] + C[j];
                for (int k=0; k<K; ++k)
                    error += P[i*K+k] * Q[j*K+k];
                loss += pow(error-T[i*M+j], 2);
                loss1 += error-T[i*M+j];
                lossabs1 += abs(error-T[i*M+j]);
                if(abs(error-T[i*M+j])>0.5) cnt++;
                // cout<<error-T[i*M+j]<<endl;
            }
        }
    }
    cout<<"test-set RMSE: "<<loss/Tlen<<"\n1-toldiff: "<<loss1<<"\n1-absdiff: "<<lossabs1/Tlen<<"\ncnt: "<<cnt<<endl<<endl;
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
    double *B = new double[N];
    double *B2 = new double[N];
    double *C = new double[M];
    double *C2 = new double[M];
    double *P = new double[N*K];
    double *P2 = new double[N*K];
    double *Q = new double[M*K];
    double *Q2 = new double[M*K];
    int *PU = new int[N*K];

    int qcnt=0;
    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            if(R[i*M+j]!=0) {
                qcnt++;
            }
    cout<<qcnt<<endl;
    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            R[i*M+j]=0;
    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            T[i*M+j]=0;

    qcnt=0;
    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            if(R[i*M+j]!=0) {
                qcnt++;
            }
    cout<<qcnt<<endl;

    for(int i=0; i<Rlen; i++) {
        R[train[i][0]*M+train[i][1]] = train[i][2];
    }
    for(int i=0; i<Tlen; i++) {
        T[test[i][0]*M+test[i][1]] = test[i][2];
    }
    qcnt=0;
    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            if(R[i*M+j]!=0) {
                qcnt++;
            }
    cout<<qcnt<<endl;

    for(int i=0; i<N; i++)
        B[i]=rand()%10/10.0;
    for(int i=0; i<M; i++)
        C[i]=rand()%10/10.0;
    for(int i=0; i<N; ++i)
        for(int j=0; j<K; ++j) {
            P[i*K+j]=rand()%10/10.0;
            PU[i*K+j]=0;
        }
    for(int i=0; i<K; ++i)
        for(int j=0; j<M; ++j)
            Q[i*M+j]=rand()%10/10.0;
            qcnt=0;
    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            if(R[i*M+j]!=0) {
                qcnt++;
            }
    cout<<qcnt<<endl;

    matrix_factorization(R,B,C,P,Q,N,M,K); // 初次进行矩阵分解得到最初的结果

    cout<<"train-set RMSE: "<<cal_rmse(R,B,C,P,Q,N,M,K,Rlen)<<endl;
    print_rst(T,B,C,P,Q,N,M,K,Tlen);

    int cnt=0,Rcnt=0;
    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            if(T[i*M+j]!=0) {
                double error = B[i] + C[j];
                for (int k=0; k<K; ++k)
                    error += P[i*K+k] * Q[j*K+k];
                Rcnt++;
                if(abs(T[i*M+j]-error) > 0.5) cnt++;
            }
    cout<<Rcnt<<endl<<cnt<<endl<<Tlen<<endl;
    cout<<(Tlen-cnt)*1.0/Tlen<<endl;


    delete [] B,B2,C,C2,P, P2, PU, Q, Q2, R, T;
    system("pause");
    return 0;
}
