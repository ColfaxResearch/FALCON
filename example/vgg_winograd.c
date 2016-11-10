#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mkl.h>
#include <string.h>
#include <hbwmalloc.h>

#include <assert.h>
#include "falcon.h"

void direct_conv(float * D0, float * F, float * O, const int N, const int K, const int P, const int Q, const int C, const int R, const int S) {
    const int P_pad = P + 2; 
    const int Q_pad = Q + 2; 
    int n, k, p, q, c, r, s; 
    float sum; 
    for (n = 0; n < N; n++) {
#pragma omp parallel for
        for (k = 0; k < K; k++) {
            for (p = 1; p < P_pad-1; p++) {
                for (q = 1; q < P_pad-1; q++) {
                    sum = 0; 
#pragma unroll
                    for (c = 0; c < C; c++) {
#pragma unroll
                        for (r = 0; r < R; r++) {
#pragma unroll
                            for (s = 0; s < S; s++) {
                                sum += F[k*C*R*S + c*R*S + r*S + s]*D0[n*C*P_pad*Q_pad + c*P_pad*Q_pad + (p+r-1)*Q_pad + (q+s-1)]; 
                            }
                        }
                    }
                    O[n*K*P*Q+ k*P*Q+ (p-1)*Q+ (q-1)] = sum; 
                }
            }
        }
    }
}


void winograd_conv(const int M, int irows, int C, int K, const int batch, long* total_flops, double* total_time, const int mod, const int verify){
   
    long i, j, n; 
    const int outHeight = irows-2; 
    const int outWidth = irows-2; 
    const int sizeI = irows*irows; 
    const int sizeF = 3*3; 
    const int sizeO = outHeight*outWidth; 
    const int tiles = (outHeight)*0.5*(outWidth)*0.5; 

    int ret; 

    float* image; 
    //allocate data on MCDRAM
    ret = hbw_posix_memalign((void*)&image, 64, batch*C*sizeI*sizeof(float)); 
    assert(image != NULL); 

    float* filter; 
    //allocate data on MCDRAM
    ret = hbw_posix_memalign((void*)&filter, 64, K*C*sizeF*sizeof(float)); 
    assert(filter != NULL); 


    float* out; 
    //allocate data on MCDRAM
    ret = hbw_posix_memalign((void*)&out, 64, batch*K*sizeO*sizeof(float)); 
    assert(out != NULL); 
    
    //initialize image in parallel
    #pragma omp parallel for private(i)
    for(i = 0; i < batch*C*sizeI; i++)
        image[i] = (float)(i%mod); 
    
    //initialize image in parallel
    #pragma omp parallel for private(i)
    for(i = 0; i < K*C*sizeF; i++)
        filter[i] = (float)(i%mod); 
    

    double timer; 
    double timer_acc = 0.0f; 

    // run for 5 iterations
    for(i = 0; i < 5; i++){
        // discard the first iteration for average timing
        if (i>0) timer= omp_get_wtime(); 
        fal_conv(M, image, irows, C, filter, K, batch, out); 
        if(i>0) timer_acc += omp_get_wtime()-timer; 
    }

    timer = timer_acc/4.0f; 
    long nflops = batch*K*C*(irows-2)*(irows-2)*3*3*2; 
    double gflops = (double) nflops*1.0e-9/timer; 
    *total_flops += nflops; 
    *total_time += timer; 

    if(verify){
        printf("Verifying WINOGRAD CONV I = %d Batch = %d C = %d K = %d \n", irows, batch, C, K); 

        float* vout; 
        //allocate data on MCDRAM
        ret = hbw_posix_memalign((void*)&vout, 64, batch*K*sizeO*sizeof(float)); 
        assert(vout != NULL); 
        direct_conv(image, filter, vout, batch, K, outHeight, outWidth, C, 3, 3); 
        for(n = 0; n < batch*sizeO*K; n++){
            if(out[n] != vout[n]){
                printf("Output Error: out[%d] = %f and vout[%d] = %f \n", n, out[n], n, vout[n]); 
                break; 
            }
        }
        hbw_free(vout); 
    }else 
        printf("WINOGRAD CONV:\tEFFECTIVE GFLOPS is %.2f \tGFlops \tand timing is \t%f  seconds \n", gflops, timer); 

    hbw_free(image); 
    hbw_free(filter); 
    hbw_free(out); 

}

int main(int argc, char** argv){


    printf("\n *****WINOGRAD CONVOLUTION******\n\n"); 
    if(argc < 2){
        printf("Enter batch_size\n"); 
        exit(-1); 
    }

    
    int i, j; 
    double timer; 
    int batch = atoi(argv[1]); 
    int verify = atoi(argv[2]); 
    

    const int max_tiles = 224*224*0.25; 
        

    const int I_array[13] = {226, 226, 114, 114, 58, 58, 58, 30, 30, 30, 16, 16, 16}; 
    const int C_array[13] = {3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512}; 
    const int K_array[13] = {64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512}; 
    const int merge_array[13] = {1, 1, 4, 4, 8, 8, 8, 16, 16, 16, 16, 16, 16}; 
    
    int t; 
    
    double total_time = 0.0f; 
    long total_flops = 0; 

    falcon_init_lib(); 

    if (verify) printf("Verifying with Reduced batch size of 8, since direct conv takes long time...\n\n\n\n"); 

    for(t = 0; t < 13; t++){
        int irows = I_array[t]; 
        int C = C_array[t]; 
        int K = K_array[t]; 
        if(verify)
            winograd_conv(1, irows, C, K, 8, &total_flops, &total_time, 50, verify); 
        else 
            winograd_conv(merge_array[t], irows, C, K, batch, &total_flops, &total_time, 50, verify); 
    }

    falcon_free_lib(); 

    
    printf("\n\n"); 
    if(!verify){
        printf("WINOGRAD: OVERALL EFFECTIVE GFLOPS is %.2f GFLops and Timing is %.4f seconds \n", (double)total_flops*1.0e-9/total_time, total_time); 
    }
    printf("\n ******************************\n\n"); 
    printf("\n\n"); 

    return 0; 


}
