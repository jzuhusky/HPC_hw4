#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include "util.h"
#include <math.h>

/**
 *
 * Note: The user will input the number of 
 * Processors/MPI_Tasks to Run and the 
 * number of Points per process for x&y dims.
 * I.e. If the User Specifies Nl = 10 and p = 4
 * Each processor will have its own mesh of 10x10 = 100 
 * grid points, for a total of 400 total grid points NOT including
 * Boundaries of Independent Processor Meshes or Overall Domain
 *
 */

// Ensure power of 4
int powerOfFour(int a){
	if (a==0)return -1;
	while(a != 1){
		if (a%4==0){a = a/4;}
		else{return -1;}
	}
	return 1;
}

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double **u, int N_l, double h){
	int i;
	double res=0.0,temp2;
	for(i=1;i<N-1;i++){
		for(j=1; j<N-1; j++){
			temp2 = (4*u[i][j]-u[i-1][j]-u[i+1][j] - u[i][j-1] - u[i][j+1] - 1.0*h*h);
			res += temp2*temp2;
		}
	}
	/* use allreduce for convenience; a reduce would also be sufficient */
	double globalRes;
	MPI_Allreduce(&res, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	return sqrt(gres);
}

int main( int argc, char *argv[]){
	int rank,p,N_l, max_iter,i,j, N_actual;
	double **uold, **unew, **temp, h, tol,res,ires;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD,&p);
	MPI_Status status1,status2,status3,status4;
	if ( argc != 3 ){
		if (rank==0)printf("Need 2 Command Line Args\n1. Size of Processor Meshes Dims\n2. Maximum # of Iterations\n");
		return -1;
	}
	if ( p!=1 && (powerOfFour(p) != 1) ){
		if (rank==0)printf("Please set number of MPI-Tasks such that #procs is a power of 4\n");
		return -1;
	} 
	// Initialize Some Global Variables
	tol = 0.0000001;
	max_iter = atoi(argv[2]);
	N_l      = atoi(argv[1]);
	N_actual = p*N_l*N_l;
	h        = 1.0/(sqrt(N_actual)+1.0);
	uold = (double**)malloc(sizeof(double*)*N_l);
	unew = (double**)malloc(sizeof(double*)*N_l);
	// Init Each Domain for Each Proc...
	for (i=0;i<N_l;i++){
                uold[i] = (double*)malloc(sizeof(double)*N_l);
                unew[i] = (double*)malloc(sizeof(double)*N_l);
        }
	int procsPerRow = sqrt(p);
	// Need some buffers...
	double *topGhost, *bottomGhost, *rightGhost, *leftGhost;
	double *topOut,*bottomOut,*rightOut,*leftOut;
	// Ghost Vectors
	topGhost     = (double*)calloc(N_l,sizeof(double));
	bottomGhost  = (double*)calloc(N_l,sizeof(double));
	rightGhost   = (double*)calloc(N_l,sizeof(double));
	leftGhost    = (double*)calloc(N_l,sizeof(double));
	//Copies of Local Edge Vectors to send out...
	topOut    = (double*)malloc(sizeof(double)*N_l);
	bottomOut = (double*)malloc(sizeof(double)*N_l);
	rightOut  = (double*)malloc(sizeof(double)*N_l);
	leftOut   = (double*)malloc(sizeof(double)*N_l);
	int iter;
	int rLeft=0,rTop=1,rRight=2,rBot=3; // When you send, give a tag for what other should receive...
	// Iterate here
	for (iter=0;iter < max_iter; iter++){
		// Populate Copies of Local Vectors to Send
		temp = uold;
		uold = unew;
		unew = temp;
		int k;
		for (k=0;k<N_l;k++){
			topOut[k]    = uold[N_l-1][k];
			bottomOut[k] = uold[0][k];
			rightOut[k]  = uold[k][N_l-1];
			leftOut[k]   = uold[k][0];
		}
		// Now Sends/Recvs
		// Swap Top and Bottom Ghost Vectors...
		if ( rank < procsPerRow ){
			MPI_Send(topOut,    N_l, MPI_DOUBLE,rank+procsPerRow, rBot, MPI_COMM_WORLD);            // Send Top
			MPI_Recv(topGhost,  N_l, MPI_DOUBLE,rank+procsPerRow, rTop, MPI_COMM_WORLD, &status1);   // Get Top
		}else if ( (rank+procsPerRow) > p-1) {
			MPI_Recv(bottomGhost,N_l, MPI_DOUBLE,rank-procsPerRow, rBot, MPI_COMM_WORLD, &status2);   // Get Bottom
			MPI_Send(bottomOut,  N_l, MPI_DOUBLE,rank-procsPerRow, rTop, MPI_COMM_WORLD);            // Send Bottom
		}else{
			MPI_Recv(bottomGhost,N_l, MPI_DOUBLE,rank-procsPerRow, rBot, MPI_COMM_WORLD, &status2);   // Get Bottom
			MPI_Send(bottomOut,  N_l, MPI_DOUBLE,rank-procsPerRow, rTop, MPI_COMM_WORLD);            // Send Bottom
			MPI_Send(topOut,    N_l, MPI_DOUBLE,rank+procsPerRow, rBot, MPI_COMM_WORLD);            // Send Top
			MPI_Recv(topGhost,  N_l, MPI_DOUBLE,rank+procsPerRow, rTop, MPI_COMM_WORLD, &status1);   // Get Top
		}
		// Swap Left and Right
		if (rank%procsPerRow == 0){
			MPI_Send(rightOut,  N_l, MPI_DOUBLE,rank+1, rLeft, MPI_COMM_WORLD);            // Send Right
			MPI_Recv(rightGhost,N_l, MPI_DOUBLE,rank+1, rRight, MPI_COMM_WORLD, &status2);   // Get Right
		}else if ((rank+1)%procsPerRow == 0){
			MPI_Recv(leftGhost,N_l, MPI_DOUBLE,rank-1, rLeft, MPI_COMM_WORLD, &status1); 
			MPI_Send(leftOut,  N_l, MPI_DOUBLE,rank-1, rRight, MPI_COMM_WORLD);
		}else{
			MPI_Recv(leftGhost,N_l, MPI_DOUBLE,rank-1, rLeft, MPI_COMM_WORLD, &status2);   
			MPI_Send(leftOut,  N_l, MPI_DOUBLE,rank-1, rRight, MPI_COMM_WORLD);          
			MPI_Send(rightOut,  N_l, MPI_DOUBLE,rank+1, rLeft, MPI_COMM_WORLD);            // Send Right
			MPI_Recv(rightGhost,N_l, MPI_DOUBLE,rank+1, rRight, MPI_COMM_WORLD, &status1);   // Get Right
		}
		// Now do Computations With Newly Received Ghost Vectors....
		for(i=1;i<N_l-1;i++){
			unew[0][i]     = (uold[0][i-1]    +uold[1][i]    +bottomGhost[i]   +uold[0][i+1]  +1.0*h*h)/4.0; // Bot Edge
			unew[i][0]     = (uold[i-1][0]    +uold[i][1]    +leftGhost[i]     +uold[i+1][0]  +1.0*h*h)/4.0; // Left Edge
			unew[N_l-1][i] = (uold[N_l-1][i-1]+uold[N_l-2][i]+uold[N_l-1][i+1] +topGhost[i]   +1.0*h*h)/4.0; // Top Edge
			unew[i][N_l-1] = (uold[i-1][N_l-1]  +uold[i][N_l-2]+uold[i+1][N_l-1] +rightGhost[i] +1.0*h*h)/4.0; // Right Edge
		}
		unew[0][0]         = (leftGhost[0]      +uold[1][0]        +bottomGhost[0]  +uold[0][1]       +1.0*h*h)/4.0;
		unew[0][N_l-1]     = (uold[0][N_l-2]    +uold[1][N_l-1]    +rightGhost[0]  +bottomGhost[N_l-1]+1.0*h*h)/4.0;
		unew[N_l-1][0]     = (uold[N_l-2][0]    +leftGhost[N_l-1]  +topGhost[0]    +uold[N_l-1][1]    +1.0*h*h)/4.0;
		unew[N_l-1][N_l-1] = (uold[N_l-2][N_l-1]+uold[N_l-1][N_l-2]+topGhost[N_l-1]+rightGhost[N_l-1] +1.0*h*h)/4.0;
		for (i=1; i<N_l-1; i++){
                        for (j=1; j < N_l-1; j++){
                                unew[i][j] = (uold[i-1][j]+uold[i+1][j] + uold[i][j-1] + uold[i][j+1] + 1.0*h*h)/4.0;
                        }
                }
		double res = compute_residual(unew,N_l,h);
		if ( iter == 0 ){
			ires = res;
		}else if ( res/ires < tol ){break;}

	}// End Iteration Loop

	int r=0;
	for (r=0;r<p;r++){
		for (i=1;i<=N_l;i++){
			for (j=1;j<=N_l;j++){
				printf("%f %f %f\n",h*((rank/procsPerRow)*N_l+i),h*((rank%procsPerRow)*N_l+j),unew[i-1][j-1]);
               		 	}
        		}	
	}


	free(topGhost);
	free(bottomGhost);
	free(rightGhost); 
	free(leftGhost); 
	free(topOut);
	free(bottomOut);
	free(rightOut);
	free(leftOut);
	for (i=0;i<N_l;i++){
                free(*(uold+i));
                free(*(unew+i));
        }
	free(uold);
	free(unew);
	MPI_Finalize();
	return 0;

}
