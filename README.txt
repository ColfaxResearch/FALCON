
The link to the paper that explains the working of FALCON in detail can be found below
http://colfaxresearch.com/falcon-library/

-------------------
Major Dependencies:
-------------------
1) ICC compiler latest
2) Intel MKL library
3) Memkind library -> https://github.com/memkind/memkind
--------------------------------------------------------

IMAGE LAYOUT : Image is a 4D data structure, image[N][C][H][W], where H=W=irows.
               W is the inner most dimension with unit stride. Image data structure 
               is stored in a linear array I[N*channels*irows*irows].

FILTER LAYOUT: Filter is a 4D data structure, filter[K][C][R][S], where R=S=3. S is 
               the inner most dimension with unit stride. Filter data structure is 
               stored in a linear array F[K*C*3*3].

OUTPUT LAYOUT: Ouput of convolution is a 4D data structure, out[N][K][oH][oW], 
               where oH=oW=(irows-2). oW is the inner most dimension with unit stride. 
               output data structure is stored in a linear array O[N*K*oH*oW].


CONV API : fal_conv(M,image,irows,C,filter,K,batch,out); 

M      -> the merge factor
image  -> pointer to I array 
irows  -> is height or width of a square image
C      -> number of image Channels
Filter -> pointer to F array 
K      -> number of filters
N      -> batch size
out    -> pointer to O array


The Merge factor provides flexibility in the way the input data layout is used. 
if M=1           -->  NCHW
else if M=N      -->  CNHW
else (1 < M < N) -->  (N/M)C(M*HW) 

Internally, FALCON reduces the convolution to a large number GEMM operations, where each 
GEMM is of the form {m,n,k},such that, m = (irows-2)(irows-2)/4 ,  k=C image channels 
and   n=K filters.

if irows << C image channels, then it results in input matrices to GEMM with very few rows 
and a large number of columns.These are skinny matrices and MKL SGEMM's performance is very 
low for such matrices.For such Input images, better performance may be obtained by merging 
M images of corresponding channels. This results in increasing the rows of the input matrices 
to form nearly square or rectangular matrices which have better performance in MKL SGEMMs. 
If the user chooses to preprocess the input in such a manner and sets M appropriately, the 
output is also obtained in a similar merged fashion where merging now happens between N and K 
output channels and this can be used as merged input for subsequent layers. This is very useful 
if the user is using multiple convolution layers chained sequentially because the user has to 
preprocess the input only once for the first layer and subsequent layers can make use of the 
merged output from the previous layer.



****** BEFORE INSTALLATION ********

-> For high performance on KNL with MCDRAM mode, Falcon uses the concept of Scratchpad memory 
   to work on,which is one time generated and used multiple times throughout the network if 
   there are multiple conv modules conneted back to back.
-> As a result the user needs to set the MAX_BATCH, MAX_IMAGE_CHANNELS, MAX_IROWS, MAX_FILTERs 
   and MAX_FILTER_CHANNELS parameters in the include/falcon.h file, which allows falcon to 
   allocate max memory that will be used by the user
-> By default these parameters are set w.r.t VGG-16 D network, which can be used as a reference 
   to understand these parameters
-> install memkind-devel, intel c compiler


*********** USAGE *****************

-> falcon_init_lib();                             //init falcon library, allocate scratch memory
-> fal_conv(M,image,irows,C,filter,K,batch,out);  // conv API
-> falcon_free_lib();                             // free up the scratch pad memory


Refer vgg_winograd.c in the example/ for usage, and to run the example program

> ./run.sh 0       ... to run 
> ./run.sh 1       ... to run and verify 


********** INSTALLATION ***********

> ./clean.sh
>./install.sh

 
