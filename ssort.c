/* Parallel sample sort
 */
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include "util.h"


static int compare(const void *a, const void *b){
  int *da = (int *)a;
  int *db = (int *)b;

  if (*da > *db)
    return 1;
  else if (*da < *db)
    return -1;
  else
    return 0;
}

int main( int argc, char *argv[]){
	int rank,numTotSamples;
	int i,N,p,s,bucketIndex=0;
	int *vec, *samples, *rootSamples,*bucketCount,*recvFrom;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD,&p);

	/* Number of random numbers per processor (this should be increased
	 * for actual tests or could be passed in through the command line */
	if (argc != 2 ){
		printf("Need 1 command line arg for Number Rands per proc\n");
		exit(1);
	}
	N = atoi(argv[1]);;
	vec = calloc(N, sizeof(int));
	/* seed random number generator differently on every core */
	srand((unsigned int) (rank + 393919));

	/* fill vector with random integers */
	for (i = 0; i < N; ++i) {
	  vec[i] = rand();	
	}
	// BEGIN TIMING
	timestamp_type time1, time2;
	get_timestamp(&time1);

	/* randomly sample s entries from vector or select local splitters,
	 * i.e., every N/P-th entry of the sorted vector */
	s = N/p;
	samples = (int*)malloc(sizeof(int)*s);
	for(i=0;i<s;i++){
		samples[i] = vec[i*(N-1)/(s-1)];
	}
	
	/* sort locally */ // After Sampling, thus samples are just as random->better distribution
  	qsort(vec, N, sizeof(int), compare);

  	/* every processor communicates the selected entries
   	* to the root processor; use for instance an MPI_Gather */

	numTotSamples = N/p;
	numTotSamples *= p;
	rootSamples = (int*)malloc(sizeof(int)*numTotSamples);
	MPI_Gather(samples,s,MPI_INT,rootSamples,s,MPI_INT,0,MPI_COMM_WORLD);

  	/* root processor does a sort, determinates splitters that
   	 * split the data into P buckets of approximately the same size */

	int buckets[p-1];/// = (int*)malloc(sizeof(int)*(p-1));
	for(i=0;i<p-1;i++){
		buckets[i] = rootSamples[i*N/p];
	}
  	qsort(buckets,p-1, sizeof(int), compare);
  	/* root process broadcasts splitters */
	MPI_Bcast(buckets,p-1,MPI_INT,0,MPI_COMM_WORLD);

	/* every processor uses the obtained splitters to decide
   	* which integers need to be sent to which other processor (local bins) */

	bucketCount = (int*)malloc(sizeof(int)*p); // p-1 splitter for p buckets
	for(i=0;i<p;i++){bucketCount[i]=0;}	 // Init BucketCounters
	// Keep In Mind, at this Point, Local Vecs are sorted!!
	i=0;
	MPI_Barrier(MPI_COMM_WORLD);
	while(i<N){
		if ( vec[i] <= buckets[bucketIndex] ){
			bucketCount[bucketIndex]++;
			i++;
		}else if(bucketIndex < p){
			bucketIndex++;
		}else{
			bucketCount[p-1]++;
			i++;
		}
	}
	recvFrom = (int*)malloc(sizeof(int)*p);
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Alltoall(bucketCount,1,MPI_INT,recvFrom,1,MPI_INT,MPI_COMM_WORLD);

	/* send and receive: either you use MPI_AlltoallV, or
	 * (and that might be easier), use an MPI_Alltoall to share
	 * with every processor how many integers it should expect,
	 * and then use MPI_Send and MPI_Recv to exchange the data */

	
	MPI_Barrier(MPI_COMM_WORLD);
	int *finalBucket;
	int totalNumbersRecv=0;
	for (i=0;i<p;i++){
		totalNumbersRecv += recvFrom[i];
	}
	finalBucket = (int*)malloc(totalNumbersRecv*sizeof(int)); // Final Receiving Buffer...
	MPI_Barrier(MPI_COMM_WORLD);

	// OK now all to all v works perfectly here
	// Since we want to send variable chunks to data
	// to Each processor from each processor
	// following:https://www.mpich.org/static/docs/v3.1/www3/MPI_Alltoallv.html

	// OK so sendBuffer is vec... Data we want to send
	// Send counts == bucketCounts vector
	// Then we need send displacements
	
	int sendDisplacements[p];
	sendDisplacements[0]=0; // For Proc 0, sending data starting at index 0
	//But for all other Procs....
	for(i=1;i<p;i++){
		//The Displacement = the Displacement of the processor before PLUS
		//The number of items the previous processor will send/recv
		sendDisplacements[i] = bucketCount[i-1]+sendDisplacements[i-1];
	}
	// Send Type = MPI_INT
	// recvCounts = recvFrom vector
	// Now need receiveDisplaements
	int recvDisplacements[p];
	recvDisplacements[0]=0; // Same logic as send
	for(i=1;i<p;i++){ // same logic as send
		//The Displacement = the Displacement of the processor before PLUS
		//The number of items the previous processor will send/recv
		recvDisplacements[i] = recvFrom[i-1]+recvDisplacements[i-1];
	}
	MPI_Alltoallv(vec,bucketCount,sendDisplacements,MPI_INT,finalBucket,recvFrom,recvDisplacements,MPI_INT,MPI_COMM_WORLD);

  	/* do a local sort */
  	qsort(finalBucket,totalNumbersRecv, sizeof(int), compare);

	// END SORTING

  	/* every processor writes its result to a file */
	char file[1024];
	snprintf(file,1024,"ssortRank%002d.dat",rank);
	FILE *filePtr = fopen(file,"w");
	for(i=0;i<totalNumbersRecv;i++){
		fprintf(filePtr,"%d\n",finalBucket[i]);
	}
	fclose(filePtr);
	get_timestamp(&time2);
	double elapsed = timestamp_diff_in_seconds(time1,time2);
	double rootE=0.0;
	MPI_Reduce(&elapsed,&rootE ,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
	if (rank==0)
        	printf("Avg Time elapsed per Processor is %f seconds.\n",rank, rootE/p);

//	int *vec, *samples, *rootSamples,*buckets,*bucketCount,*recvFrom;
	free(finalBucket);free(vec);free(samples);free(rootSamples);free(bucketCount);free(recvFrom);
	MPI_Finalize();
	return 0;
}
