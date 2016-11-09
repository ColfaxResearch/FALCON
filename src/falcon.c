#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mkl.h>
#include <string.h>
#include <hbwmalloc.h>
#include <assert.h>
#include <falcon.h>

const long MAX_TILES = (MAX_IROWS-2)*(MAX_IROWS-2)*0.25; 
// STRIDE is the max image*C*batch for image
const long STRIDE = (MAX_BATCH)*(MAX_IMAGE_CHANNELS+18)*(MAX_TILES+13); 
// FSTRIDE is the max C*K for filter
const long FSTRIDE = (MAX_FILTER_CHANNELS+1)*(MAX_FILTERS+1); 

float* t_filter;    
float* t_image;    
float* c_out;    

// setup scratch memory used in the algorithm
void falcon_init_lib(){


    int ret; 
    ret = hbw_posix_memalign((void*)&t_filter, 64, 16*FSTRIDE*sizeof(float));    
    assert(t_filter != NULL); 
    ret = hbw_posix_memalign((void*)&t_image, 64, 16*STRIDE*sizeof(float));    
    assert(t_image != NULL); 
    ret = hbw_posix_memalign((void*)&c_out, 64, 16*STRIDE*sizeof(float));    
    assert(c_out != NULL); 

}

// free up the scratch pad
void falcon_free_lib(){

    hbw_free(t_filter);    
    hbw_free(t_image);    
    hbw_free(c_out);    
}


// INTERNAL FUNCTION : FORM MATRIX A from input data, also includes transformation
static void get_tiles(const float* restrict image, const int ldi, const int irows, const int sizeI, const int C, float* restrict otile, const int N, const int ntiles){
   
    int t, u; 
    #pragma omp parallel for 
    for(t = 0; t < N*C; t++){
            
        int i, j, x; 
        float tmp[16] __attribute__((aligned(64))); 
        float s[16] __attribute__((aligned(64))); 

        const float* data = image+t*sizeI; 
        int tile_count = t*ntiles; 
        
        // work on one image plane at a time, irrespective of the order
        for(i = 0; i < irows-2; i += 2){
            #pragma unroll(4)            
            for(j = 0; j < (irows-2); j += 2){
                tmp[0 :4] =data[(i+0)*ldi+j:4]; 
                tmp[4 :4] =data[(i+1)*ldi+j:4]; 
                tmp[8 :4] =data[(i+2)*ldi+j:4]; 
                tmp[12:4] =data[(i+3)*ldi+j:4]; 

                // The tranformation manually simplified
                s[0 ] =(tmp[0] - tmp[8 ]) - (tmp[2 ]- tmp[10]);   
                s[1 ] =(tmp[1] - tmp[9 ]) + (tmp[2 ]- tmp[10]); 
                s[2 ] =(tmp[2] - tmp[10]) - (tmp[1 ]- tmp[9 ]); 
                s[3 ] =(tmp[1] - tmp[9 ]) - (tmp[3 ]- tmp[11]); 
                s[4 ] =(tmp[4] + tmp[8 ]) - (tmp[6 ]+ tmp[10]); 
                s[5 ] =(tmp[5] + tmp[9 ]) + (tmp[6 ]+ tmp[10]); 
                s[6 ] =(tmp[6] + tmp[10]) - (tmp[5 ]+ tmp[9 ]); 
                s[7 ] =(tmp[5] + tmp[9 ]) - (tmp[7 ]+ tmp[11]); 
                s[8 ] =(tmp[8] - tmp[4 ]) - (tmp[10]- tmp[6 ]); 
                s[9 ] =(tmp[9] - tmp[5 ]) + (tmp[10]- tmp[6 ]); 
                s[10] =(tmp[10]- tmp[6 ]) - (tmp[9 ]- tmp[5 ]); 
                s[11] =(tmp[9] - tmp[5 ]) - (tmp[11]- tmp[7 ]); 
                s[12] =(tmp[4] - tmp[12]) - (tmp[6 ]- tmp[14]); 
                s[13] =(tmp[5] - tmp[13]) + (tmp[6 ]- tmp[14]); 
                s[14] =(tmp[6] - tmp[14]) - (tmp[5 ]- tmp[13]); 
                s[15] =(tmp[5] - tmp[13]) - (tmp[7 ]- tmp[15]); 

                // manually unrolled scatter to get max performance
                otile[tile_count+0*STRIDE ] = s[0 ]; 
                otile[tile_count+1*STRIDE ] = s[1 ]; 
                otile[tile_count+2*STRIDE ] = s[2 ]; 
                otile[tile_count+3*STRIDE ] = s[3 ]; 
                otile[tile_count+4*STRIDE ] = s[4 ]; 
                otile[tile_count+5*STRIDE ] = s[5 ]; 
                otile[tile_count+6*STRIDE ] = s[6 ]; 
                otile[tile_count+7*STRIDE ] = s[7 ]; 
                otile[tile_count+8*STRIDE ] = s[8 ]; 
                otile[tile_count+9*STRIDE ] = s[9 ]; 
                otile[tile_count+10*STRIDE] = s[10]; 
                otile[tile_count+11*STRIDE] = s[11]; 
                otile[tile_count+12*STRIDE] = s[12]; 
                otile[tile_count+13*STRIDE] = s[13]; 
                otile[tile_count+14*STRIDE] = s[14]; 
                otile[tile_count+15*STRIDE] = s[15]; 


                tile_count++; 
            }
        }
    }

}

// INTERNAL FUNCTION: FORM MATRIX B, also includes filter transform
static void filter_transform(const float* restrict filter, const int C, const int K, float* restrict out){

    int m, n, x; 
    const float *F; 

    #pragma omp parallel for collapse(2) private(m, n, x, F)
    for(m = 0; m < K; m++){
        for(n = 0; n < C; n++){
            float c1[16] __attribute__((aligned(64))); 
            F = filter+n*3*3 + m*3*3*C; 

            // work on in 3x3 plane at a time
            // The tranformation manually simplified
            c1[0]  = F[0]; 
            c1[1]  = (F[0]+F[2]+F[1])*0.5f; 
            c1[2]  = (F[0]+F[2]-F[1])*0.5f; 
            c1[3]  = F[2]; 
            c1[4]  = (F[0]+F[6]+F[3])*0.5f; 
            c1[5]  = ((F[0]+F[6]+F[3])+(F[2]+F[8]+F[5])+(F[1]+F[7]+F[4]))*0.25f; 
            c1[6]  = ((F[0]+F[6]+F[3])+(F[2]+F[8]+F[5])-(F[1]+F[7]+F[4]))*0.25f; 
            c1[7]  = (F[2]+F[8]+F[5])*0.5f; 
            c1[8]  = (F[0]+F[6]-F[3])*0.5f; 
            c1[9]  = ((F[0]+F[6]-F[3])+(F[2]+F[8]-F[5])+(F[1]+F[7]-F[4]))*0.25f; 
            c1[10] = ((F[0]+F[6]-F[3])+(F[2]+F[8]-F[5])-(F[1]+F[7]-F[4]))*0.25f; 
            c1[11] = (F[2]+F[8]-F[5])*0.5f; 
            c1[12] = F[6]; 
            c1[13] = (F[6]+F[8]+F[7])*0.5f; 
            c1[14] = (F[6]+F[8]-F[7])*0.5f; 
            c1[15] = F[8]; 

            // scatter
            #pragma unroll(16)
            for(x = 0; x < 16; x++){
                out[x*FSTRIDE+m*C+n] = c1[x]; 
            }
        }
    }
}

// INTERNAL FUNCTION
// GEMM specific to Ist layer of VGG with (M, N, K) = (12544, 64, 3)
// MKL performs bad
static void gemm_ker(int m, int n, int k, const float* a, const int lda, const float* b, const int ldb, float* c, const int ldc){

    const int BLK = 16; 
    int x, xx, y, z, i; 

    for(z = 0; z < n; z++){
        for(x = 0; x < m; x += BLK){
            float p[BLK] __attribute__((aligned(64))); 
            p[0:BLK] = 0.0f; 
            #pragma unroll(3)
            for(y = 0; y < 3; y++){
                #pragma vector aligned
                for(i = 0; i < BLK; i++){
                    p[i] += a[x+i+y*lda]*b[y+z*ldb]; 
                }
            }
            c[x+z*ldc:BLK] = p[0:BLK]; 
        }
    }

}


// INTERNAL FUNCTION
// C = A*B with beta = 0.0f and alpha = 1.0f
// Number of gemm calls is 16*BATCH 
static void batched_gemm(const float* restrict image, const int irows, const int icols, const float* restrict filter, const int frows, const int fcols, float* restrict  out, const int batch){

    int t, i; 
    const char trans ='n'; 
    const float alpha = 1.0; 
    const float beta =  0.0; 
    const int ldi = irows; 
    const int ldf = frows; 
    const int ldo = irows; 
    
    #pragma omp parallel for collapse(2) private(t, i)
    for(i = 0; i < 16; i++){
        for(t = 0; t < batch; t++){
            const float* im = image+i*STRIDE+t*irows*icols; 
            const float* fi = filter+i*FSTRIDE; 
            float* ot = out+i*STRIDE+t*irows*fcols; 
            if(icols == 3) gemm_ker(irows, fcols, icols, im, ldi, fi, ldf, ot, ldo); 
            else sgemm(&trans, &trans, &irows, &fcols, &icols, &alpha, im, &ldi, fi, &ldf, &beta, ot, &ldo); 
        }
    }

} 



static void out_transform(const float* restrict d, const int K, const int ntiles, float* restrict out, const int ldo, const int oH, const  int oW, const int N){
    
    int t; 
    int sizeO = oH*oW; 
    
    #pragma omp parallel for 
    for(t = 0; t < N*K; t++){
        
        float c1[16] __attribute__((aligned(64))); 
        float temp[8] __attribute__((aligned(64))); 
        float c2[4] __attribute__((aligned(64))); 
        
        float* data = out +t*sizeO; 
        int tile_offset = t*ntiles; 
        
        int i, j;    
        // work on one output plane at a time, irrespective of the order
        for(i = 0; i < oH; i += 2){
            for(j = 0; j < oW; j += 2){
                
                // gather the 16 elements form C to form a tile
                c1[0 ] = d[tile_offset+0 *STRIDE]; 
                c1[1 ] = d[tile_offset+1 *STRIDE]; 
                c1[2 ] = d[tile_offset+2 *STRIDE]; 
                c1[3 ] = d[tile_offset+3 *STRIDE]; 
                c1[4 ] = d[tile_offset+4 *STRIDE]; 
                c1[5 ] = d[tile_offset+5 *STRIDE]; 
                c1[6 ] = d[tile_offset+6 *STRIDE]; 
                c1[7 ] = d[tile_offset+7 *STRIDE]; 
                c1[8 ] = d[tile_offset+8 *STRIDE]; 
                c1[9 ] = d[tile_offset+9 *STRIDE]; 
                c1[10] = d[tile_offset+10*STRIDE]; 
                c1[11] = d[tile_offset+11*STRIDE]; 
                c1[12] = d[tile_offset+12*STRIDE]; 
                c1[13] = d[tile_offset+13*STRIDE]; 
                c1[14] = d[tile_offset+14*STRIDE]; 
                c1[15] = d[tile_offset+15*STRIDE]; 

                // The tranformation manually simplified
                temp[0] = c1[0]+c1[1]+ c1[2]; 
                temp[1] = c1[1]-c1[2]- c1[3]; 
                temp[2] = c1[4]+c1[5]+ c1[6]; 
                temp[3] = c1[5]-c1[6]- c1[7]; 
                temp[4] = c1[8]+c1[9]+ c1[10]; 
                temp[5] = c1[9]-c1[10]- c1[11]; 
                temp[6] = c1[12]+c1[13]+ c1[14]; 
                temp[7] = c1[13]-c1[14]- c1[15]; 

                c2[0] = temp[0]+temp[2]+temp[4]; 
                c2[1] = temp[1]+temp[3]+temp[5]; 
                c2[2] = temp[2]-temp[4]-temp[6]; 
                c2[3] = temp[3]-temp[5]-temp[7]; 
                
                data[i*ldo+j]  =c2[0];     
                data[i*ldo+j+1]  =c2[1]; 
                data[(i+1)*ldo+j] = c2[2]; 
                data[(i+1)*ldo+j+1] = c2[3];     
                tile_offset++; 
            }
        }
    }
}


void fal_conv(const int M, float* restrict image, const int irows, const int C, float* restrict filter, const int K, const int batch, float* restrict out){

    const int outHeight = irows-2; 
    const int outWidth = irows-2; 
    const int sizeI = irows*irows; 
    const int tiles = (outHeight)*0.5*(outWidth)*0.5; 
        
    filter_transform(filter, C, K, t_filter); 
    get_tiles(image, irows, irows, sizeI, C, t_image, batch, tiles); 
    batched_gemm(t_image, M*tiles, C, t_filter, C, K, c_out, batch/M); 
    out_transform(c_out, K, tiles, out, outWidth, outHeight, outWidth, batch); 

}

