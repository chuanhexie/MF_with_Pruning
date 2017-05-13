#include<iostream>
#include<iomanip>
#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<fstream>
#include<sstream>
#include<time.h>
using namespace std;
#define feature_size 7
#define datasize 8000
#define STEP 1000
#define STEPR 50
#define ALAPA 0.0002
#define BETA 0.1

int Rlen=0,Tlen=0;

void matrix_factorization(const double *R,double *P,double *Q,int N,int M,int K,int steps=STEP,float alpha=ALAPA,float beta=BETA) {
    for(int step =0; step<steps; ++step) {
        for(int i=0; i<N; ++i) {
            for(int j=0; j<M; ++j) {
                if(R[i*M+j]>0) {
                    double error = R[i*M+j];
                    for(int k=0; k<K; ++k) error -= P[i*K+k]*Q[k*M+j]; //
                    for(int k=0; k<K; ++k) {
                        P[i*K+k] += alpha * (2 * error * Q[k*M+j] - beta * P[i*K+k]);//
                        Q[k*M+j] += alpha * (2 * error * P[i*K+k] - beta * Q[k*M+j]);
                        /*
                            P(i,j) += alpha * e(i,j) * Q(k,j) - beta * P(i,j)
                            Q(i,j) += alpha * e(i,j) * P(k,j) - beta * Q(i,j)
                        */
                    }
                }
            }
        }
        double loss=0;
        for(int i=0; i<N; ++i) {
            for(int j=0; j<M; ++j) {
                if(R[i*M+j]>0) {
                    double error = 0;
                    for(int k=0; k<K; ++k)
                        error += P[i*K+k]*Q[k*M+j];
                    loss += pow(R[i*M+j]-error,2);
                    for(int k=0; k<K; ++k)
                        loss += (beta/2) * (pow(P[i*K+k],2) + pow(Q[k*M+j],2));
                }
            }
        }

        if(loss<datasize*0.01 || step == STEP-1) {
            cout<<"Final loss:"<<loss/Rlen<<endl;
            break;
        }
        /*else if (step%(int)(0.1*STEP)==0)
            cout<<"loss:"<<loss/Rlen<<endl;*/
    }
}

void matrix_recounstruction(const double *R,double *P,double *Q,int N,int M,int K,int steps=STEPR,float alpha=ALAPA,float beta=BETA) {
    for(int step =0; step<steps; ++step) { // 重建与初次的差异步骤不大，但需要判断为0的位置不进行更新
        for(int i=0; i<N; ++i) {
            for(int j=0; j<M; ++j) {
                if(R[i*M+j]>0) {
                    double error = R[i*M+j];
                    for(int k=0; k<K; ++k) error -= P[i*K+k]*Q[k*M+j]; //
                    for(int k=0; k<K; ++k)
                        if(P[i*K+k]!=0) {
                            P[i*K+k] += alpha * (2 * error * Q[k*M+j] - beta * P[i*K+k]);//
                            Q[k*M+j] += alpha * (2 * error * P[i*K+k] - beta * Q[k*M+j]);
                        }
                }
            }
        }
    }
}

double cal_rmse(double *R,double *P,double *Q,int N,int M,int K) {
    double loss = 0;
    for(int i=0; i<N; ++i) {
        for(int j=0; j<M; ++j) {
            if(R[i*M+j] > 0 ) {
                double error=0;
                for (int k=0; k<K; ++k)
                    error+=P[i*K+k]*Q[k*M+j];
                loss += pow(error-R[i*M+j], 2);
            }
        }
    }
    return loss;
}

int main() {
    ifstream infile;
    infile.open("train_small.txt", ios::in);
    if (!infile.is_open()) cout<<"NO\n";
    stringstream ss;
    char buffer[256];
    int a,b,c;
    int train[datasize][3], test[datasize][3];
    int MAXa=0, MAXb=0, MAXc=0, MINc=-1;

    int seed = time(0);
    // cout<<seed<<endl;
    srand(seed);
    while(infile.getline(buffer,256) && buffer!=NULL) {
        // cout<<buffer<<endl;
        ss << buffer;
        ss>>a>>b>>c;
        ss.clear();
        //cout<<a<<' '<<b<<' '<<c<<endl;
        if (rand()%10>1) { //
            train[Rlen][0] = a;
            train[Rlen][1] = b;
            train[Rlen][2] = c;
            Rlen++;
        }
        else {
            test[Tlen][0] = a;
            test[Tlen][1] = b;
            test[Tlen][2] = c;
            Tlen++;
        }
        if (a>MAXa) MAXa = a;
        if (b>MAXb) MAXb = b;
        if (c>MAXc) MAXc = c;
        if (c<MINc || MINc==-1) MINc = c;
    }
    infile.close();
    int N=MAXa+1, M=MAXb+1, K=feature_size;
    cout<<N<<' '<<M<<' '<<K<<"\n";
    cout<<Rlen<<' '<<Tlen<<endl;

    double *R = new double[N*M];
    double *T = new double[N*M];
    double *P = new double[N*K];
    double *P2 = new double[N*K];
    int *PU = new int[N*K];
    double *Q = new double[M*K];
    double *Q2 = new double[M*K];

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
            P[i*K+j]=rand()%2;
            PU[i*K+j]=0;
        }
    for(int i=0; i<K; ++i)
        for(int j=0; j<M; ++j)
            Q[i*M+j]=rand()%2;

    matrix_factorization(R,P,Q,N,M,K); // 初次进行矩阵分解得到最初的结果
    
    double loss = 0, loss1 = 0, lossabs1 = 0, lossr = 0;
    //cout<<"train-set first-rst RMSE: "<<cal_rmse(R,P,Q,N,M,K)/Rlen<<endl;double loss = 0, loss1 = 0, lossabs1 = 0;
    for(int i=0; i<N; ++i) {
        for(int j=0; j<M; ++j) {
            if(T[i*M+j] > 0 ) {
                double error=0;
                for (int k=0; k<K; ++k)
                    error+=P[i*K+k]*Q[k*M+j];
                /*if (error>MAXc){
                    error = MAXc;
                }else if (error<MINc){
                    error = MINc;
                }*/
                loss += pow(error-T[i*M+j], 2);
                loss1 += error-T[i*M+j];
                lossabs1 += abs(error-T[i*M+j]);
                // cout<<(int)abs(error-T[i*M+j])<<' '<<error<<' '<<T[i*M+j]<<endl;
            }
            /*else if(R[i*M+j] > 0) {
                double error = 0;
                for(int k=0; k<K; ++k)
                    error+=P[i*K+k]*Q[k*M+j];
                lossr += pow(R[i*M+j]-error,2);
            }*/
        }
    }


    lossr=cal_rmse(R,P,Q,N,M,K);
    cout<<"train-set first-rst RMSE: "<<lossr/Rlen<<endl;
    cout<<"test-set  first-rst RMSE: "<<loss/Tlen<<"\n1-toldiff: "<<loss1<<"\n1-absdiff: "<<lossabs1/Tlen<<endl;

    /*cout<<"\n\n\norignal P P P P\n";
    for(int i=0; i<N; ++i) {
        cout<<(i/1000)%10<<(i/100)%10<<(i/10)%10<<i%10<<": ";
        for(int j=0; j<K; ++j)
            cout<<setprecision(4)<<(char)((P[i*K+j]<0)?7:32)<<fixed<<P[i*K+j]<<' ';
        cout<<endl;
    }*/

    int gg=0,cnt=0;
    lossr = 0;
    while(gg++<N*K/2) {
        lossr=cal_rmse(R,P,Q,N,M,K);
        int min=10,x=0,y=0;
        for(int i=0; i<N; i++)
            for(int j=0; j<K; j++) {
                P2[i*K+j] = P[i*K+j];
                if(P[i*K+j]!=0 && !PU[i*K+j] && min>abs(P[i*K+j])) {// 寻找P中未重复出现过的最小值位置
                    min=P[i*K+j];
                    x=i;
                    y=j;
                }
            }
        for(int i=0; i<M; i++)
            for(int j=0; j<K; j++) {
                Q2[i*K+j] = Q[i*K+j];/*
                if(P[i*K+j]!=0 && !PU[i*K+j] && min>abs(P[i*K+j])) {// 寻找Q中未重复出现过的最小值位置
                    min=Q[i*K+j];
                    x=i;
                    y=j;
                }*/
            }
        if(abs(min)>0.8) break;
        P2[x*K+y] = 0;
        PU[x*K+y] = 1;
        matrix_recounstruction(R,P2,Q2,N,M,K); // 将修改后的P2代入去迭代
        if(cal_rmse(R,P2,Q,N,M,K)<lossr*1.05) { // 观察rmse变化情况，若增加太多则不改变
            for(int q=0; q<N*K; q++) P[q]=P2[q];
            for(int q=0; q<M*K; q++) Q[q]=Q2[q];
            cnt++;
        }
    }
    cout<<N*K<<' '<<gg<<' '<<cnt<<endl;

    // double loss = 0, loss1 = 0, lossabs1 = 0;
    loss = 0; loss1 = 0; lossabs1 = 0; lossr = 0;
    for(int i=0; i<N; ++i) {
        for(int j=0; j<M; ++j) {
            if(T[i*M+j] > 0 ) {
                double error=0;
                for (int k=0; k<K; ++k)
                    error+=P[i*K+k]*Q[k*M+j];
                /*if (error>MAXc){
                    error = MAXc;
                }else if (error<MINc){
                    error = MINc;
                }*/
                loss += pow(error-T[i*M+j], 2);
                loss1 += error-T[i*M+j];
                lossabs1 += abs(error-T[i*M+j]);
                // cout<<(int)abs(error-T[i*M+j])<<' '<<error<<' '<<T[i*M+j]<<endl;
            }
            /*else if(R[i*M+j] > 0) {
                double error = 0;
                for(int k=0; k<K; ++k)
                    error+=P[i*K+k]*Q[k*M+j];
                lossr += pow(R[i*M+j]-error,2);
            }*/
        }
    }


    lossr=cal_rmse(R,P,Q,N,M,K);
    cout<<"train-set RMSE: "<<lossr/Rlen<<endl;
    cout<<"test-set  RMSE: "<<loss/Tlen<<"\n1-toldiff: "<<loss1<<"\n1-absdiff: "<<lossabs1/Tlen<<endl;

    /*cout<<"\n\n\nP P P P\n";
    for(int i=0; i<N; ++i) {
        cout<<(i/1000)%10<<(i/100)%10<<(i/10)%10<<i%10<<": ";
        for(int j=0; j<K; ++j)
            cout<<setprecision(4)<<(char)((P[i*K+j]<0)?7:32)<<fixed<<P[i*K+j]<<' ';
        cout<<endl;
    }

    cout<<"\n\n\nQ Q Q Q\n";
    for(int i=0; i<M; ++i) {
        cout<<(i/1000)%10<<(i/100)%10<<(i/10)%10<<i%10<<": ";
        for(int j=0; j<K; ++j)
            cout<<setprecision(4)<<(char)((Q[i*K+j]<0)?7:32)<<fixed<<Q[i*K+j]<<' ';
        cout<<endl;
    }*/
    /*cout<<"\n\n\nQ Q Q Q\n";
    for(int j=0; j<M; ++j) {
        for(int i=0; i<K; ++i)
            cout<<Q[i*M+j]<<' ';
        cout<<endl;
    }*/

    delete [] P, P2, PU, Q, Q2, R, T;

    /*int cnt4pu=0;
    for(int i=0;i<N*K;i++) if(PU[i]==1) cnt4pu++;
    cout<<gg<<' '<<cnt4pu<<endl;
    int qa,qb;
    while(cin>>qa>>qb){
        cout<<PU[qa*K+qb]<<endl;
    }*/
    system("pause");
    return 0;
}
