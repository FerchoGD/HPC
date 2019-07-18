#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "../common/common.h"
#include <cuda_runtime.h>
#include <fstream.h>
#include <omp.h>

using namespace std;

void generatedata(int *traf, int size){
    time_t t;
    srand((unsigned)time(&t));

    for (int i=0; i<size; i++){
        traf[i]=(int)(rand()%2);
    }
}

void simulationonhost(int *A,int size,int iter){
    
    //Malloc para el vector donde voy a guardar el resultado
    int iteraciones=0;

    while(iteraciones<iter){
        int *new_traffic = new int[size];
        bool change_ult=false;

        for (int i=0; i<size-1;i++){
            if(A[i]==1){
                if(A[i+1]==0){
                    new_traffic[i]=0;
                    new_traffic[i+1]=1;
                    cout<<"Hola ";
                    i++;
                    if(i>=size-1)change_ult=true;                
                }
                else{
                new_traffic[i]=A[i];
                cout<<" Da igual ";
                }
            }
            else{
                new_traffic[i]=A[i];
                cout<<" Da igual ";
                }
             
            
        }
        //Comprobando último elemento

        if(A[0]==0 && A[size-1]==1){
            new_traffic[0]=1;
            new_traffic[size-1]=0;
            cout<<" Hola ult ";
            change_ult=true;
        }
        if(!change_ult){
            new_traffic[size-1]=A[size-1];
            cout<<"Da igual ult ";
        }        
        cout<<"Iteración #"<<iteraciones<<" : "<<endl;
        for(int j=0;j<size;j++){
            A[j]=new_traffic[j];
            cout<<A[j]<<" ";
        }
        cout<<endl;
        iteraciones++;
        delete new_traffic;
    }
    
}

__global__ void simulationongpu(int *A,const int size, int iter){
    int i= threadIdx.x;
    if(i<iter) {
        int iteraciones=0;
    
        int *new_traffic = new int[size];
        bool change_ult=false;

        for (int i=0; i<size-1;i++){
            if(A[i]==1){
                if(A[i+1]==0){
                    new_traffic[i]=0;
                    new_traffic[i+1]=1;
                    cout<<"Hola ";
                    i++;
                    if(i>=size-1)change_ult=true;                
                }
                else{
                new_traffic[i]=A[i];
                cout<<" Da igual ";
                }
            }
            else{
                new_traffic[i]=A[i];
                cout<<" Da igual ";
                }
            
            
        }
        //Comprobando último elemento

        if(A[0]==0 && A[size-1]==1){
            new_traffic[0]=1;
            new_traffic[size-1]=0;
            cout<<" Hola ult ";
            change_ult=true;
        }
        if(!change_ult){
            new_traffic[size-1]=A[size-1];
            cout<<"Da igual ult ";
        }        
        cout<<"Iteración #"<<iteraciones<<" : "<<endl;
        for(int j=0;j<size;j++){
            A[j]=new_traffic[j];
            cout<<A[j]<<" ";
        }
        cout<<endl;
        iteraciones++;
        delete new_traffic;
    
    }
}

int main(){
    srand(time(NULL));
    int tam=10;
    int *vehiculos = new int[tam];
    int simulations=5;
    generatedata(vehiculos,tam);
    cout<<"Inicial : ";
    for(int ind=0; ind<tam; ind++){
        cout<<vehiculos[ind]<<" ";
    }
    cout<<endl;
    ofstream archivo("resultados.txt");

    //Configurando CUDA

    








    //Simulación en Secuencial
    cout<<"Simulating Traffic: "<<endl;
    double start = omp_get_wtime();
    simulationonhost(vehiculos,tam,simulations);
    double finish = omp_get_wtime( );
    double time = finish - start;
    fs<<time<<"-"<<tam<<"-"<<simulations<<"\n";

    //Simulación en Paralelo
    start= omp_get_wtime();
    simulationongpu(vehiculos,tam,simulations);
    finish = omp_get_wtime();
    time = finish - start;
    fs<<"Cuda: "<<tiempo<<"-"<<tam<<"-"<<simulations<<"\n";



    archivo.close();
    delete vehiculos;
}