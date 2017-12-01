/*TRUONG DAI HOC BACH KHOA HA NOI - HANOI UNIVERSITY OF SCIENCE AND TECHNOLOGY*/
/*VIEN CONG NGHE THONG TIN VA TRUYEN THONG - SCHOOL OF INFORMATION AND COMMUNICATION TECHNOLOGY*/

/*import libraries*/
#include <stdlib.h>
#include <stdio.h>
#include <conio.h>
#include <time.h>
#include <cuda.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#pragma comment(lib, "cudart")

/*set USE_CPU = 1 to compute in CPU*/
/*set USE_GPU = 0 to compute in GPU*/

#define USE_CPU 1
#define USE_GPU 0

/*define macro I2D: index of elements in 2D-grid*/
#define I2D(ni, i, j) ((i) + (ni)*(j))

/*compute temperature or diffusion concentration on CPU*/
void cpu_DiffusionEquation(int ni, 
						   int nj,
						   float dt, 
						   float *temp_in,
						   float *temp_out) {
    int i, j, i00, im10, ip10, i0m1, i0p1;
	float dx = 0.1; 
	float dy = 0.1;
	float td = 1.0;						/*td: thermal diffusity - he so khuech tan nhiet*/
    float d2tdx2, d2tdy2;

    /*spatial dicretization - roi rac hoa theo khong gian*/
    /*iterate through all the points in computing domain (except margin points)*/
    for (j=1; j < nj-1; j++) {
        for (i=1; i < ni-1; i++) {

	    /*indexes of center and neighborhoods I2D*/
            i00 = I2D(ni, i, j);
            im10 = I2D(ni, i-1, j);
            ip10 = I2D(ni, i+1, j);
            i0m1 = I2D(ni, i, j-1);
            i0p1 = I2D(ni, i, j+1);

	    /*calculate derivates*/
            d2tdx2 = (temp_in[im10] - 2*temp_in[i00] + temp_in[ip10])/(dx*dx);
            d2tdy2 = (temp_in[i0m1] - 2*temp_in[i00] + temp_in[i0p1])/(dy*dy);
		
	    /*time integration - tich hop theo thoi gian*/
	    /*update temperature in center point*/
            temp_out[i00] = temp_in[i00] + dt*td*(d2tdx2 + d2tdy2);
        }
    }
}

/*kernel funtion compute temperature/diffusion concentration on GPU */
__global__ void gpu_DiffusionEquation(int ni, 
                                      int nj,
                                      float dt,
                                      float *temp_in,
                                      float *temp_out) {
    int i, j, i00, im10, ip10, i0m1, i0p1;
    float dx = 0.1; 
	float dy = 0.1;
	float td = 1.0;
	float d2tdx2, d2tdy2;
	
    /*the indexes i and j of threads*/
    i = blockIdx.x*blockDim.x + threadIdx.x;
    j = blockIdx.y*blockDim.y + threadIdx.y;

    /*indexes of center and neighborhoods I2D*/
    i00 = I2D(ni, i, j);
    im10 = I2D(ni, i-1, j);
    ip10 = I2D(ni, i+1, j);
    i0m1 = I2D(ni, i, j-1);
    i0p1 = I2D(ni, i, j+1);
    
    /*check threads in computating domain (except margin points)*/
    if (i > 0 && i < ni-1 && j > 0 && j < nj-1) {
            /*spatial discretization*/
	    /*calculate derivates*/
            d2tdx2 = (temp_in[im10] - 2*temp_in[i00] + temp_in[ip10])/(dx*dx);
            d2tdy2 = (temp_in[i0m1] - 2*temp_in[i00] + temp_in[i0p1])/(dy*dy);
   	             
	    /*time integration*/
	    temp_out[i00] = temp_in[i00] + dt*td*(d2tdx2 + d2tdy2);        
    }
}

int main(int argc, char *argv[]) 
{
    int ni, nj, nstep;
    float *h_temp1, *h_temp2,  *d_temp1, *d_temp2, *tmp_temp;
    int i, j, i2d, istep;
    float temp_bl, temp_br, temp_tl, temp_tr;
	float dt = 0.1;

	int Grid_Dim = 32;
	int Block_Dim = 32;

	clock_t startclock, stopclock;
	double elapsedtime;  
   										
	cudaEvent_t start, stop;				/*use cuda events to measure performance*/
	float elapsed_time;	

    FILE *fp;										
   
    /*define the size of grid and time step*/
    ni = 2048;
    nj = 2048;
    nstep = 10000;
    
    /*allocate memory on host*/
    h_temp1 = (float *)malloc(sizeof(float)*ni*nj);
    h_temp2 = (float *)malloc(sizeof(float)*ni*nj);

    /*data sample*/
    for (i=1; i < ni-1; i++) {
        for (j=1; j < nj-1; j++) {
            i2d = j + ni*i;
            h_temp1[i2d] = 10;
        }
    }
    
    /*initialize values for margins*/
    /*bl-bottom left br-bottom right tl-top left tr-top right*/
    temp_bl = 200.0f;
    temp_br = 500.0f;
    temp_tl = 200.0f;
    temp_tr = 500.0f;

    /*set temperature of points in edges*/
    for (i=0; i < ni; i++) {
        /*bottom (temp_bl = 200.0f, temp_br = 500.0f)*/
        j = 0;
        i2d = i + ni*j;
        h_temp1[i2d] = temp_bl + (temp_br-temp_bl)*(float)i/(float)(ni-1);

        /*top (temp_tl = 200.0f, temp_tr = 500.0f)*/
        j = nj-1;
        i2d = i + ni*j;
        h_temp1[i2d] = temp_tl + (temp_tr-temp_tl)*(float)i/(float)(ni-1);
    }

    for (j=0; j < nj; j++) {
        /*left (temp_bl = 200.0f, temp_tl = 200.0f)*/
        i = 0;
        i2d = i + ni*j;
        h_temp1[i2d] = temp_bl + (temp_tl-temp_bl)*(float)j/(float)(nj-1);

        /*right (temp_br = 500.0f, temp_tr = 500.0f)*/
        i = ni-1;
        i2d = i + ni*j;
	    h_temp1[i2d] = temp_br + (temp_tr-temp_br)*(float)j/(float)(nj-1);
    }

	/*copy data from h_temp1 to h_temp2*/
	cudaMemcpy(h_temp2, h_temp1, sizeof(float)*ni*nj, cudaMemcpyHostToHost);
	      
	/*allocate array for storing temperature on device*/
    cudaMalloc((void **)&d_temp1, sizeof(float)*ni*nj);
    cudaMalloc((void **)&d_temp2, sizeof(float)*ni*nj);

	/*transfer data from host to device*/
    cudaMemcpy((void *)d_temp1, (void *)h_temp1, sizeof(float)*ni*nj, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_temp2, (void *)h_temp1, sizeof(float)*ni*nj, cudaMemcpyHostToDevice);
    
    
//    cudaEventCreate(&start);					                /*measure performance with <cuda_runtime.h>*/
//	  cudaEventCreate(&stop);
//    cudaEventRecord(start, 0);

	startclock = clock();							/*measure performance with <time.h>*/

    for (istep=0; istep < nstep; istep++) {
        //printf("%i\n", istep);
		if (USE_CPU == 1) {                                           
            cpu_DiffusionEquation(ni, nj, dt, h_temp1, h_temp2);
	    
	    // change pointers
            tmp_temp = h_temp1;
            h_temp1 = h_temp2;
            h_temp2 = tmp_temp;
            
        }
		if (USE_GPU == 1) {
	    
			// define architure of grid and block 
			dim3 Grid(Grid_Dim, Grid_Dim);			
			dim3 Block(Block_Dim, Block_Dim);					           
            
	    /*invoke kernel for GPU computing*/ 
            gpu_DiffusionEquation<<<Grid, Block>>>(ni, nj, dt, d_temp1, d_temp2);
            /*change pointers*/						     
            tmp_temp = d_temp1;
            d_temp1 = d_temp2;
            d_temp2 = tmp_temp;
        }    
    } 

	stopclock = clock();
	elapsedtime = ((double)(stopclock-startclock))/CLOCKS_PER_SEC;
	printf("Time execution: %f s.\n", elapsedtime);

//  cudaThreadSynchronize();
//	cudaEventRecord(stop, 0);									//mesure end time
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&elapsed_time, start, stop);
//  printf("\nTime execution: %f ms.\n", elapsed_time);

	/*copy data from GPU to CPU*/
	if (USE_CPU == 0) {
        cudaMemcpy((void *)h_temp1, (void *)d_temp1, sizeof(float)*ni*nj, cudaMemcpyDeviceToHost);
    }
    
	/*save data*/
	if(USE_CPU == 1){
		fp = fopen("cpu_output.dat", "w");
		fprintf(fp, "%i %i", ni, nj);
		for(j=0; j<nj; j++){
			for(i=0; i<ni; i++){
				fprintf(fp, "%f\n", h_temp1[i+ni*j]);
			}
		}
		fclose(fp);
	}
	else if (USE_GPU == 1){
		fp = fopen("gpu_output.dat", "w");
		fprintf(fp, "%i %i\n", ni, nj);
		for (j=0; j < nj; j++) {
			for (i=0; i < ni; i++) {
				fprintf(fp, "%f\n", h_temp1[i+ni*j]);
			}
		}
		fclose(fp);
	}
	
	/*free memory GPU va CPU*/
	free(h_temp1);
	free(h_temp2);
	cudaFree(d_temp1);
	cudaFree(d_temp2);

//	cudaEventDestroy(start);
//	cudaEventDestroy(stop);

	return 0;
	getch();
}
