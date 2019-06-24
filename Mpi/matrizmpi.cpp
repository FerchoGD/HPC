#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <random>
#include <chrono>

using namespace std;

void printMatrix(int *M, int S[2]){
    /*
    M is a Matrix
    S[0] -> M rows
    S[1] -> M cols
    Prints M into the console
    */
    for(int row = 0; row < S[0]; row++){
        for(int col = 0; col < S[1]; col++){
            cout<<setw(8)<<*(M + col + (row * S[1]))<<" ";
        }
        cout<<endl;
    }
}

void fillMatrix(int *M, int S[2], uniform_int_distribution<int> &dist, default_random_engine &re){
    /*
    M is a Matrix
    S[0] -> M rows
    S[1] -> M cols
    Fills M with random integer numbers between 1 and 999
    */
    for(int row = 0; row < S[0]; row++){
        for(int col = 0; col < S[1]; col++){
            *(M + col + (row * S[1])) = dist(re);
        }
    }
}

void multiplyMatrices(int *M1, int *M2, int *MR, int S1[2], int S2[2], int SR[2]){
    /*
    M1, M2, and MR are Matrices
    SX[0] -> MX rows
    SX[0] -> MX cols
    Multiplies M1 and M2, and saves the result on MR
    */
    for(int row = 0; row < SR[0]; row++){
        for(int col = 0; col < SR[1]; col++){
            *(MR + col + (row * SR[1])) = 0;
            for(int z = 0; z < S1[1]; z++){
                *(MR + col + (row * SR[1])) += ((*(M1 + z + (row * S1[1]))) * (*(M2 + col + (z * S2[1]))));
            }
        }
    }
}

int main(int argc, char** argv){

    MPI_Init(NULL, NULL);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3){
        if( rank == 0){
            cout<<" ERROR: Parameters"<<endl;
            cout<<" USAGE: ./<executable> <matriz_size> <print_matrices>"<<endl;
            cout<<"      | matriz_size    = 5 --> 5x5 Matrix"<<endl;
            cout<<"      | print_matrices = 0 --> Do NOT Print Matrices"<<endl;
            cout<<"      | print_matrices = 1 --> Do Print Matrices"<<endl;
        }
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        system("clear");

        uniform_int_distribution<int> distribution(1, 9);
        default_random_engine random_engine;
        random_engine.seed(time(NULL));

        int N = stoi(argv[1]);
        int printMatrices = stoi(argv[2]);

        cout<<" RUNNING: Matrix Multiplication --> "<<N<<"x"<<N<<endl<<endl;

        int SA[2] = {N, N};
        int SB[2] = {N, N};
        int SC[2] = {SA[0], SB[1]};

        int *A = new int[SA[0] * SA[1]];
        int *B = new int[SB[0] * SB[1]];
        int *C = new int[SC[0] * SC[1]];

        fillMatrix(A, SA, distribution, random_engine);  
        fillMatrix(B, SB, distribution, random_engine);

        int workers = size - 1;
        int rowsPerWorker = N / workers;
        int rowsToDistribute = N % workers;
        
        int Offset = 0;

        int SD[2];

        auto start = chrono::high_resolution_clock::now();

        for(int worker = 1; worker < size; worker++){

            if (rowsToDistribute > 0){
                SD[0] = rowsPerWorker + 1;
                SD[1] = N;
                rowsToDistribute--;
            }else{
                SD[0] = rowsPerWorker;
                SD[1] = N;
            }      
            //cout<<worker<<" ";  

            MPI_Send(SD, 2, MPI_INT, worker, 0, MPI_COMM_WORLD);

            MPI_Send(SB, 2, MPI_INT, worker, 0, MPI_COMM_WORLD);

            MPI_Send(A + Offset, SD[0]*SD[1], MPI_INT, worker, 0, MPI_COMM_WORLD);

            MPI_Send(B, SB[0]*SB[1], MPI_INT, worker, 0, MPI_COMM_WORLD);

            Offset += SD[0]*SD[1];
        }

        Offset = 0;

        for(int worker = 1; worker < size; worker++){
            
            int SD[2];
            MPI_Recv(SD, 2, MPI_INT, worker, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Recv(C + Offset, SD[0]*SD[1], MPI_INT, worker, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            Offset += SD[0]*SD[1];
        }

        auto finish = chrono::high_resolution_clock::now();
        chrono::duration<double> running_time = finish - start;

        if (printMatrices){
            cout<<" Matrix A   --> "<<SA[0]<<"x"<<SA[1]<<endl;
            printMatrix(A, SA);
            cout<<" Matrix B   --> "<<SB[0]<<"x"<<SB[1]<<endl;
            printMatrix(B, SB);
            cout<<" Matrix AxB --> "<<SC[0]<<"x"<<SC[1]<<endl;
            printMatrix(C, SC);
            cout<<endl;
        }

        cout<<" Running Time --> "<<setprecision(9)<<fixed<<running_time.count()<<" seconds"<< endl;

        delete[] A;
        delete[] B;
        delete[] C;

    } else {

        int SA[2];
        int SB[2];

        MPI_Recv(SA, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Recv(SB, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int SC[2] = {SA[0], SB[1]};

        int *A = new int[SA[0] * SA[1]];
        int *B = new int[SB[0] * SB[1]]; 
        int *C = new int[SC[0] * SC[1]];

        MPI_Recv(A, SA[0]*SA[1], MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Recv(B, SB[0]*SB[1], MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        multiplyMatrices(A, B, C, SA, SB, SC);

        MPI_Send(SC, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(C, SC[0]*SC[1], MPI_INT, 0, 0, MPI_COMM_WORLD);

        delete[] A;
        delete[] B;
        delete[] C;
    }

    MPI_Finalize();

    return 0;
}
