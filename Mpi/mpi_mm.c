#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 10
#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

int main(int argc, char *argv[])
{
    int numtasks,              /* number of tasks in partition */
        taskid,                /* matrix1 task identifier */
        numworkers,            /* number of worker tasks */
        source,                /* task id of message source */
        dest,                  /* task id of message destination */
        rows,                  /* rows of matrix A sent to each worker */
        averow, extra, offset, /* used to determine rows sent to each worker */
        i, j, k;               /* misc */

    double matrix1[SIZE][SIZE],        /* matrix A to be multiplied */
        matrix2[SIZE][SIZE],           /* matrix B to be multiplied */
        matrixR[SIZE][SIZE];           /* result matrix C */

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    printf("Numtasks: %d \n", numtasks);

    if (numtasks < 2)
    {
        int errCod;
        printf("Need at least two MPI tasks...\n");
        MPI_Abort(MPI_COMM_WORLD, errCod);
        exit(1);
    }

    numworkers = numtasks - 1;

    if (taskid == MASTER)
    {
        printf("MPI MM has started with %d tasks.\n", numtasks);

        for (i=0; i < SIZE; i++)
            for (j=0; j < SIZE; j++)
                matrix1[i][j] = ((double)rand() / (double)(RAND_MAX)) * 9.0;

        for (i=0; i < SIZE; i++)
            for (j=0; j < SIZE; j++)
                matrix2[i][j] = ((double)rand() / (double)(RAND_MAX)) * 9.0;

        /* Measure start time */
        double start = MPI_Wtime();

        /* Send matrix data to the worker tasks */
        averow = SIZE / numworkers;
        extra = SIZE % numworkers;
        offset = 0;
        for (dest = 1; dest <= numworkers; dest++)
        {
            rows = (dest <= extra) ? averow + 1 : averow;
            printf("Sending %d rows to task %d offset=%d\n",rows,dest,offset);
            MPI_Send(&offset, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&matrix1[offset][0], rows * SIZE, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&matrix2, SIZE * SIZE, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD);
            offset = offset + rows;
        }

        /* Receive results from worker tasks */
        for (i = 1; i <= numworkers; i++)
        {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(&matrixR[offset][0], rows * SIZE, MPI_DOUBLE, source, FROM_WORKER, MPI_COMM_WORLD, &status);
            printf("Received results from task %d\n",source);
        }

        /* Measure finish time */
        double finish = MPI_Wtime();
        printf("Done in %f seconds.\n", finish - start);
    }

    /**************************** worker task ************************************/
    if (taskid > MASTER)
    {
        printf("Task %d start running.\n", taskid);
        MPI_Recv(&offset, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&matrix1, rows * SIZE, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&matrix2, SIZE * SIZE, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);

	//#pragma omp parallel for private(i,j,k)
        for (k = 0; k < SIZE; k++)
            for (i = 0; i < rows; i++)
            {
                matrixR[i][k] = 0.0;
                for (j = 0; j < SIZE; j++)
                    matrixR[i][k] = matrixR[i][k] + matrix1[i][j] * matrix2[j][k];
            }

        MPI_Send(&offset, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&matrixR, rows * SIZE, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}
