
CC=mpicc
flags=-O3
EXEC=int_ring.run
EXECarray=array_ring.run

METHODS=int_ring array_ring

all:${METHODS}

int_ring: int_ring.c
	${CC} ${FLAGS} -o ${EXEC} ${ompFlag} $^
array_ring: array_ring.c
	${CC} ${FLAGS} -o ${EXECarray} ${ompFlag} $^

clean:
	rm -f *.run
