/********************************************************
 * Kernels to be optimized for the CS:APP Performance Lab
 ********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "defs.h"
/*
 * Please fill in the following team_t struct
 */
team_t team = {

        "e2547651",      /* First student ID */
        "Aneliya Abdimalik",       /* First student name */

};


/********************
 * NORMALIZATION KERNEL
 ********************/

/****************************************************************
 * Your different versions of the normalization functions go here
 ***************************************************************/

/*
 * naive_normalize - The naive baseline version of convolution
 */
char naive_normalize_descr[] = "naive_normalize: Naive baseline implementation";
void naive_normalize(int dim, float *src, float *dst) {
    float min, max;
    min = src[0];
    max = src[0];
	int i,j;
    for ( i = 0; i < dim; i++) {
        for ( j = 0; j < dim; j++) {
	
            if (src[RIDX(i, j, dim)] < min) {
                min = src[RIDX(i, j, dim)];
            }
            if (src[RIDX(i, j, dim)] > max) {
                max = src[RIDX(i, j, dim)];
            }
        }
    }

    for ( i = 0; i < dim; i++) {
        for ( j = 0; j < dim; j++) {
            dst[RIDX(i, j, dim)] = (src[RIDX(i, j, dim)] - min) / (max - min);
        }
    }
}

/*
 * normalize - Your current working version of normalization
 * IMPORTANT: This is the version you will be graded on
 */
char normalize_descr[] = "Normalize: Current working version";
void normalize(int dim, float *src, float *dst)
{ float low = src[0];
    float high = src[0];
    int i,j;

    /* Find the minimum and maximum values in the array*/
    for (i = 0; i < dim; i++) {
        int idx=i*dim;
        for (j = 0; j < dim; j += 4) {
            float val1 = src[idx+j];
            float val2 = src[idx+j+1];
            float val3 = src[idx+j+2];
            float val4 = src[idx+j+3];

            
            low= (val1<low)?val1:low;
            low= (val2<low)?val2:low;
            low= (val3<low)?val3:low;
            low= (val4<low)?val4:low;
            high=(val1 > high)?val1:high;
            high=(val2 > high)?val2:high;
            high=(val3 > high)?val3:high;
            high=(val4 > high)?val4:high;
        }
    }

    float scale = 1.0f / (high - low);

    for (i = 0; i < dim; i++) {
        int idx=i*dim;
        for (j = 0; j < dim; j++) {
            float value = src[idx+j];
            dst[idx+j] = (value - low) * scale;
        }
    }
}


/*********************************************************************
 * register_normalize_functions - Register all of your different versions
 *     of the normalization functions  with the driver by calling the
 *     add_normalize_function() for each test function. When you run the
 *     driver program, it will test and report the performance of each
 *     registered test function.
 *********************************************************************/

void register_normalize_functions() {
    add_normalize_function(&naive_normalize, naive_normalize_descr);
    add_normalize_function(&normalize, normalize_descr);
    /* ... Register additional test functions here */
}




/************************
 * KRONECKER PRODUCT KERNEL
 ************************/

/********************************************************************
 * Your different versions of the kronecker product functions go here
 *******************************************************************/

/*
 * naive_kronecker_product - The naive baseline version of k-hop neighbours
 */
char naive_kronecker_product_descr[] = "Naive Kronecker Product: Naive baseline implementation";
void naive_kronecker_product(int dim1, int dim2, float *mat1, float *mat2, float *prod) {
   int i,j,k,l;
 for ( i = 0; i < dim1; i++) {
        for ( j = 0; j < dim1; j++) {
            for ( k = 0; k < dim2; k++) {
                for ( l = 0; l < dim2; l++) {
                    prod[RIDX(i, k, dim2) * (dim1 * dim2) + RIDX(j, l, dim2)] = mat1[RIDX(i, j, dim1)] * mat2[RIDX(k, l, dim2)];
                }
            }
        }
    }
}



/*
 * kronecker_product - Your current working version of kronecker_product
 * IMPORTANT: This is the version you will be graded on
 */
char kronecker_product_descr[] = "Kronecker Product: Current working version";
void kronecker_product(int dim1, int dim2, float *mat1, float *mat2, float *prod)
{
int total_dim = dim1 * dim2;
int i,j,k,l;

    for (i = 0; i < dim1; i += 2) {
	int s1=i*dim1;
	int s2=(i+1)*dim1;
        for (j = 0; j < dim1; j += 2) {
            float a = mat1[s1+j];
            float b = mat1[s1+j+1];
            float c = mat1[s2+j];
            float d = mat1[s2+j+1];

            int row_start1 = i * dim2;
            int row_start2 = (i + 1) * dim2;

            for (k = 0; k < dim2; k += 2) {
                int idx1 = (row_start1 + k)* total_dim;
                int idx2 = (row_start1 + k + 1)*total_dim;
                int idx3 = (row_start2 + k)*total_dim;
                int idx4 = (row_start2 + k + 1)*total_dim;
		int t1=k*dim2;
		int t2=(k+1)*dim2;
                for (l = 0; l < dim2; l += 2) {
                    float p = mat2[t1+l];
                    float q = mat2[t1+l+1];
                    float r = mat2[t2+l];
                    float s = mat2[t2+l+1];
			int q=j*dim2+l;
                    int idx_base1 = idx1 + q;
                    int idx_base2 = idx2 + q;
                    int idx_base3 = idx3 + q;
                    int idx_base4 = idx4 + q;

                    prod[idx_base1] = a * p;
                    prod[idx_base1 + 1] = a * q;
                    prod[idx_base2] = a * r;
                    prod[idx_base2 + 1] = a * s;

                    prod[idx_base1 + dim2] = b * p;
                    prod[idx_base1 + dim2 + 1] = b * q;
                    prod[idx_base2 + dim2] = b * r;
                    prod[idx_base2 + dim2 + 1] = b * s;

                    prod[idx_base3] = c * p;
                    prod[idx_base3 + 1] = c * q;
                    prod[idx_base4] = c * r;
                    prod[idx_base4 + 1] = c * s;

                    prod[idx_base3 + dim2] = d * p;
                    prod[idx_base3 + dim2 + 1] = d * q;
                    prod[idx_base4 + dim2] = d * r;
                    prod[idx_base4 + dim2 + 1] = d * s;
                }
            }
        }
    }    
}

/******************************************************************************
 * register_kronecker_product_functions - Register all of your different versions
 *     of the kronecker_product with the driver by calling the
 *     add_kronecker_product_function() for each test function. When you run the
 *     driver program, it will test and report the performance of each
 *     registered test function.  
 ******************************************************************************/

void register_kronecker_product_functions() {
    add_kronecker_product_function(&naive_kronecker_product, naive_kronecker_product_descr);
    add_kronecker_product_function(&kronecker_product, kronecker_product_descr);
    /* ... Register additional test functions here */
}

