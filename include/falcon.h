
// The below parameters are required to generate the scratch pad memory
// It is required to reserve enough memory to store data for all the sizes you will be working on
// For example : by default, the below parameters are set for the VGG-16 Type D network 
#define MAX_BATCH           64
#define MAX_IMAGE_CHANNELS  64
#define MAX_IROWS           226
#define MAX_FILTER_CHANNELS 512
#define MAX_FILTERS         512


void fal_conv(
	      const int M, 
	      float* restrict image, 
	      const int irows, 
	      const int C, 
	      float* restrict filter, 
	      const int K, 
	      const int N, 
	      float* restrict out);
void falcon_init_lib();
void falcon_free_lib();
// IMAGE LAYOUT : Image is a 4D data structure, image[N][C][H][W], where H=W=irows.
//                W is the inner most dimension with unit stride. Image data structure is stored in a linear
//                array I[N*channels*irows*irows].

// FILTER LAYOUT: Filter is a 4D data structure, filter[K][C][R][S], where R=S=3. S is the inner most dimension
//                with unit stride. Filter data structure is stored in a linear array F[K*C*3*3].

// OUTPUT LAYOUT: Ouput of convolution is a 4D data structure, out[N][K][oH][oW], where oH=oW=(irows-2).
//                oW is the inner most dimension with unit stride. output data structure is stored in a linear
//                array O[N*K*oH*oW].


// M      -> the merge factor
// image  -> pointer to I array 
// irows  -> is height or width of a square image
// C      -> number of image Channels
// Filter -> pointer to F array 
// K      -> number of filters
// N      -> batch size
// out    -> pointer to O array


// The Merge factor provides flexibility in the way the input data layout is used. 
// if M=1           -->  NCHW
// else if M=N      -->  CNHW
// else (1 < M < N) -->  (N/M)C(M*HW) 

// Internally, FALCON reduces the convolution to a large number GEMM operations, where each GEMM is of the form {m,n,k}, 
// such that, m = (irows-2)(irows-2)/4 ,  k=C image channels and   n=K filters.

// if irows << C image channels, then it results in input matrices to GEMM with very few rows and a large number of columns. 
// These are skinny matrices and MKL SGEMM's performance is very low for such matrices.
// For such Input images, better performance may be obtained by merging M images of corresponding channels. This results in increasing the rows of the 
// input matrices to form nearly square or rectangular matrices which have better performance in MKL SGEMMs. If the user chooses to preprocess the input 
// in such a manner and sets M appropriately, the output is also obtained in a similar merged fashion where merging now happens between N and K output channels
// and this can be used as merged input for subsequent layers. This is very useful if the user is using multiple convolution layers chained sequentially because the 
// user has to preprocess the input only once for the first layer and subsequent layers can make use of the merged output from the previous layer.


