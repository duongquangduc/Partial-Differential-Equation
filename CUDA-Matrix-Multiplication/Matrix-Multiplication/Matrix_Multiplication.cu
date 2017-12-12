/*TRUONG DAI HOC BACH KHOA HA NOI - HANOI UNIVERSITY OF SCIENCE AND TECHNOLOGY*/
/*VIEN CONG NGHE THONG TIN VA TRUYEN THONG - SCHOOL OF INFORMATION AND COMMUNICATION TECHNOLOGY*/

------Matrix Multiplication in CUDA Programming------
-----1 Thread computes 1 element of matrix product-----

#include <stdio.h>
#include <conio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#pragma comment(lib, "cudart")


void cpu_matrixMul(int *a, int *b, int *c, int N){

	int row, col, k, sum;
	
	for (row=0; row<N; row++)
		for (col=0; col<N; col++){
			sum = 0;
			for (k=0; k<N; k++)
				sum += a[row*N+k]*b[k*N+col];	
			c[row*N+col]=sum;
		}
}


__global__ void gpu_matrixMul1(int *a, int*b, int *c, int N){

	int col = threadIdx.x;
	int row = threadIdx.y;
	int k, sum = 0;
	
	if ((row < N) && (col <N)){			
		for (k = 0; k < N; k++){
				sum += a[row*N+k]*b[k*N+col];
				c[row*N+col] = sum;
		}			
	}
}


__global__ void gpu_matrixMul(int *a, int *b, int *c, int N){

	int col = threadIdx.x + blockIdx.x*blockDim.x;
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int k, sum = 0;

	if ((row < N) && (col <N)){			
			for (k = 0; k < N; k++){
					sum += a[row*N+k]*b[k*N+col];
					c[row*N+col] = sum;
			}			
	}
}

int main (int argc, char *argv[]){
	
									
	char key;
	
	int Grid_Dim = 1;				
	int Block_Dim = 1;				

	int N=10;						
	
	int *a, *b, *c, *d;
	int *dev_a, *dev_b, *dev_c;
	int size;						

									
	cudaEvent_t start, stop;
	float elapsed_time_ms;

/*----------------------------------------INPUT DATA----------------------------------------*/
printf("||--------------------MATRIX MULTIPLICATION--------------------||\n");
do{
	printf("DEVICE SPECIFICATIONS --- COMPUTING CAPABILITY: 2.1\n");
	printf("\n");
	printf("||----------------------INPUT PARAMETERS AND DATA----------------------||\n");

	printf("Input size of square matrix, the numbers of dimension is %d: ", N);
	scanf("%d", &N);

	do{
		printf("\nInput the numbers of threads in each dimension x/y in a block, currently %d: ", Block_Dim);
		scanf("%d", &Block_Dim);

		printf("\nInput the number of block in a grid in x/y dimesions, currently %d: ", Grid_Dim);
		scanf("%d", &Grid_Dim);

		if (Block_Dim>32)
			printf("ERROR!!! The number of thread in a block is exceedind, please input a valid number.\n"); 
		if ((Grid_Dim*Block_Dim)<N)
			printf("ERROR!!! The number of thread is less than the number of elements in matrix, please input again.\n");
		else
			printf("Number of threads not used = %d\n", ((Grid_Dim*Block_Dim)-N)*((Grid_Dim*Block_Dim)-N));
		
	}while((Block_Dim > 32) || (Block_Dim*Grid_Dim < N));

	dim3 Grid(Grid_Dim, Grid_Dim);			//define grid struture
	dim3 Block(Block_Dim, Block_Dim);		//define block structure

	size = N*N*sizeof(int);					//the size of matrix (bytes)
	
	a=(int*)malloc(size);					
	b=(int*)malloc(size);	
	c=(int*)malloc(size);					
	d=(int*)malloc(size);					


//Initializing some data
for(int i= 0; i<N; i++)
	for(int j=0; j<N; j++){
		a[i*N+j]=j;
		b[i*N+j]=j*1;
	}

printf("||----------------------------PRINT MATRIX----------------------------||\n");
printf("\nMATRIX A AND B:\n");
	for (int i=0; i<N; i++){
		for(int j=0; j<N; j++)
			printf("%d ", a[i*N+j]);				
			printf("\n");
	}
	
/*--------------------COMPUTING ON GPU--------------------*/
//allocate memory on device
cudaMalloc((void**)&dev_a, size);
cudaMalloc((void**)&dev_b, size);
cudaMalloc((void**)&dev_c, size);

//transfer data from host to device
cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
cudaMemcpy(dev_c, c, size, cudaMemcpyHostToDevice);

cudaEventCreate(&start);									
cudaEventCreate(&stop);

cudaEventRecord(start, 0);

gpu_matrixMul<<<Grid, Block>>>(dev_a, dev_b, dev_c, N);		

cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);			

cudaEventRecord(stop, 0);									
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsed_time_ms, start, stop);
printf("\nThoi gian tinh toan tren GPU: %f ms.\n", elapsed_time_ms);

/*--------------------COMPUTING ON CPU--------------------*/
cudaEventRecord(start, 0);									

cpu_matrixMul(a, b, d, N);									

cudaEventRecord(stop, 0);									
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsed_time_ms, start, stop);

printf("\nTime Elapsed on CPU: %f ms.\n",elapsed_time_ms );

/*--------------------COMPARE RESULTS ON HOST AND DEVICE--------------------*/
	for(int i=0; i<N*N; i++){
	if(c[i]!= d[i])
		printf("\nERROR!!! CPU AND GPU COMPUTE DIFFERENT RESULTS\n");
	else
		printf("\nCORRECT!!! CPU AND GPU COMPUTE SAME RESULTS\n");
		break;
	}

	printf("\nMATRIX OUTPUT:\n");
	for (int i=0; i<N; i++){
		for(int j=0; j<N; j++)
			printf("%d ", c[i*N+j]);				
			printf("\n");
	}
	
	printf("\nINPUT N TO BEGIN NEW COMPUTATION:\n");
	scanf("%c", &key);
	scanf("%c", &key);

	}while (key=='n');									
	
/*--------------------FREE MEMORY--------------------*/
free(a);
free(b);
free(c);
cudaFree(dev_a);
cudaFree(dev_b);
cudaFree(dev_c);

cudaEventDestroy(start);
cudaEventDestroy(stop);

return 0;
getch();

}
