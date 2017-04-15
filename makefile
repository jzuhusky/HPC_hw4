
CC=mpicc
flags=-O3

METHODS=2DLaplace_MPI ssort

all:${METHODS}

2DLaplace_MPI: 2DLaplace_MPI.c
	${CC} ${FLAGS} $^ -o Laplace.run -lm

ssort: ssort.c
	${CC} ${FLAGS} $^ -o ssort.run -lm
clean:
	rm -f *.run
