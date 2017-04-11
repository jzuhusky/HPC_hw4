QUICK-NOTE: The image labeled DataRate.png
-> The units are incorrect (only slightly)
-> The data for MB/sec is too large by a factor of 2!!
-> When running the code, I forgot to account
-> for the fact that I was recording the time taken
-> to send data over and back. Therefore, the max dataRate
-> between crunchy1/3 maxes out around 1GByte/sec

Joe Zuhusky:

HW3 MPI Submission

See data plots attached for Data collected

TripTimeVsMB.ps ->
Shows trip time of sending an array between 2 machines and back

DataRate.png -> Show how bandwidth becomes saturated as the 
number of trips around the loop increases. This is pretty much 
what was expected.

Call 'make' to compile

mpirun {INPUT YOUR OWN FLAGS HERE} ./int_ring.run #loops
mpirun {INPUT YOUR OWN FLAGS HERE} ./int_array.run #loops #sizeOfIntegerArrayToSendAboutTheLoop


Final project pitch:

I'll be experimenting with parallel implementations of the
Multigrid Method for 2D Laplace/Possion eq (maybe 3D if I have time
but I understand that this is not far from the 2D case).

First I plan on implementing a serial version. 
Then exploring what kind of speedup is achieveable using openMP,MPI & definitely GPUs/Cuda.
I would definitely like to deploy some code to Stampede and see
how much speedup is available in that environment. 

I do not have a partner for this project. I put a notice on piazza, but no one has reached out to me.
I havent joined someone else's team since I feel I have a lot to learn 
about multigrid methods (and I want to study this topic!!). I plan on doing this project alone as of now,
which is entirely fine with me. I am a good C programmer (MSCS student). 
So I feel like the only hurdle will be familiarizing myself with multigrid and
parallelization techniques (which is the point!). 











