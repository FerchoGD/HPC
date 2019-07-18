#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <ctime>
//#include <cuda_runtime.h>
#include <fstream>
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
                    i++;
                    if(i>=size-1)change_ult=true;                
                }
                else{
                new_traffic[i]=A[i];
                }
            }
            else{
                new_traffic[i]=A[i];
                }
             
            
        }
        //Comprobando último elemento

        if(A[0]==0 && A[size-1]==1){
            new_traffic[0]=1;
            new_traffic[size-1]=0;
            change_ult=true;
        }
        if(!change_ult){
            new_traffic[size-1]=A[size-1];
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
    //Host Memory
    int *vehiculos = new int[tam];
    int simulations=1000000;


    //Vamos a iterar sobre las simulaciones y tamano
    generatedata(vehiculos,tam);
    cout<<"Inicial : ";
    for(int ind=0; ind<tam; ind++){
        cout<<vehiculos[ind]<<" ";
    }
    cout<<endl;
    ofstream archivo("resultados.txt");

    //Configurando CUDA
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    size_t nBytes = tam *sizeof(int);

    int *gpuref=(int *)malloc(nBytes);
    int *d_A=(int *)malloc(nBytes);

    memset(gpuref,0,nBytes);

    //Malloc for GPU

    CHECK(cudaMalloc((int**)&vehiculos, nBytes));
    CHECK(cudaMalloc((int**)&gpuref, nBytes));

    //Transfer Data From Host to Device
    CHECK(cudaMemcpy(d_A,vehiculos, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_A, gpuRef, nBytes, cudaMemcpyHostToDevice));

    dim3 block (tam);
    dim3 grid (1);

    simulationongpu<<grid,block>>(d_A,tam,simulaciones);

    //Copy Device To Host

    CHECK(cudaMemcpy(gpuref,d_A,nBytes,cudaMemcpyDeviceToHost));

    //Free

    CHECK(cudaFree(d_A));

    


    //Simulación en Secuencial
    cout<<"Simulating Traffic: "<<endl;
    unsigned t0,t1;
    t0=clock();
    simulationonhost(vehiculos,tam,simulations);
    t1=clock();

    double time = (double(t1-t0)/CLOCKS_PER_SEC);
    archivo<<time<<"-"<<tam<<"-"<<simulations<<"\n";

    //Simulación en Paralelo
    /*
    unsigned start,finish;
    start= clock();
    simulationongpu(vehiculos,tam,simulations);
    finish = clock();
    time = (double(t1-t0)/CLOCKS_PER_SEC);
    fs<<"Cuda: "<<tiempo<<"-"<<tam<<"-"<<simulations<<"\n";
    */



    archivo.close();
    //Free Host
    delete vehiculos;

    CHECK(cudaDeviceReset());
    return 0;
}