/*TRUONG DAI HOC BACH KHOA HA NOI*/
/*VIEN CONG NGHE THONG TIN VA TRUYEN THONG*/
/*DO AN TOT NGHIEP NGANH CONG NGHE THONG TIN*/
/*NGHIEN CUU-CAI DAT-UNG DUNG BXL DO HOA DA DUNG GPGPU TRONG TINH TOAN HIEU NANG CAO*/
/*THUC HIEN: SV. DUONG QUANG DUC*/
/*HUONG DAN: TS. VU VAN THIEU*/
/*GIAI PHUONG TRINH KHUECH TAN - PHUONG TRINH TRUYEN NHIET TREN GPU*/
/*TU KHOA: GPU COMPUTING, GPGPU, CUDA, PARALLEL COMPUTING, HEAT EQUATION*/

#include <stdlib.h>
#include <stdio.h>
#include <conio.h>
#include <time.h>
#include <cuda.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#pragma comment(lib, "cudart")

/*dat USE_CPU = 1 thuc hien tinh toan tren CPU*/
/*dat USE_GPU = 1 thuc hien tinh toan tren GPU*/

#define USE_CPU 1
#define USE_GPU 0

/*dinh nghia macro chi thi I2D xac dinh chi so cac phan tu tren luoi 2 chieu*/
#define I2D(ni, i, j) ((i) + (ni)*(j))

/*ham thuc hien tinh nhiet do/nong do khuech tan - thuc hien tren CPU*/
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

	/*roi rac hoa theo khong gian - spatial dicretization*/
    /*lap tren tat ca cac diem cua mien tinh toan (ngoai tru cac diem bien)*/
    for (j=1; j < nj-1; j++) {
        for (i=1; i < ni-1; i++) {

		/*cac chi cua cac diem trung tam va cac diem lan can - cong thuc tinh theo macro I2D*/
            i00 = I2D(ni, i, j);
            im10 = I2D(ni, i-1, j);
            ip10 = I2D(ni, i+1, j);
            i0m1 = I2D(ni, i, j-1);
            i0p1 = I2D(ni, i, j+1);

	    /*tinh cac dao ham*/
            d2tdx2 = (temp_in[im10] - 2*temp_in[i00] + temp_in[ip10])/(dx*dx);
            d2tdy2 = (temp_in[i0m1] - 2*temp_in[i00] + temp_in[i0p1])/(dy*dy);
		
		/*tich hop theo thoi gian - time integration*/
	    /*cap nhat nhiet do tai diem trung tam*/
            temp_out[i00] = temp_in[i00] + dt*td*(d2tdx2 + d2tdy2);
        }
    }
}

/*ham kernel thuc hien tinh nhiet do/nong do khuech tan - thuc hien tren GPU*/
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
	
	/*cac chi so i va j cua cac thread*/
	i = blockIdx.x*blockDim.x + threadIdx.x;
    j = blockIdx.y*blockDim.y + threadIdx.y;

	/*cac chi cua cac diem trung tam va cac diem lan can - cong thuc tinh theo macro I2D*/
    i00 = I2D(ni, i, j);
    im10 = I2D(ni, i-1, j);
    ip10 = I2D(ni, i+1, j);
    i0m1 = I2D(ni, i, j-1);
    i0p1 = I2D(ni, i, j+1);
    
	/*kiem tra cac thread trong vung tinh toan (khong tinh cac diem tren bien hoac ben ngoai)*/
    if (i > 0 && i < ni-1 && j > 0 && j < nj-1) {
            /*roi rac hoa theo khong gian - spatial discretization*/
			/*tinh cac dao ham*/
            d2tdx2 = (temp_in[im10] - 2*temp_in[i00] + temp_in[ip10])/(dx*dx);
            d2tdy2 = (temp_in[i0m1] - 2*temp_in[i00] + temp_in[i0p1])/(dy*dy);
   	             
			/*tich hop theo thoi gian - time integration*/
			/*tinh cac nhiet do/nong do khuech tan tai moi diem*/
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
   										
	cudaEvent_t start, stop;						/*su dung cuda events do thoi gian thuc hien*/
	float elapsed_time;	

    FILE *fp;										/*ghi ket qua ra file*/
   
	/*dinh nghia kich thuoc luoi diem tinh toan va so buoc thoi gian tich hop*/
    ni = 2048;
    nj = 2048;
    nstep = 10000;
    
	/*cap phat bo nho mang nhiet do tren host*/
    h_temp1 = (float *)malloc(sizeof(float)*ni*nj);
    h_temp2 = (float *)malloc(sizeof(float)*ni*nj);

 	/*khoi tao gia tri ban dau cho cac diem trong vung tinh toan*/
	/*data sample*/
    for (i=1; i < ni-1; i++) {
        for (j=1; j < nj-1; j++) {
            i2d = j + ni*i;
            h_temp1[i2d] = 10;
        }
    }
    
	/*khoi tao du lieu tai cac diem bien - cac diem goc canh*/
	/*bl-bottom left br-bottom right tl-top left tr-top right*/
    temp_bl = 200.0f;
    temp_br = 500.0f;
    temp_tl = 200.0f;
    temp_tr = 500.0f;

	/*thiet lap nhiet do cac diem tren cac canh thong qua cac diem goc bang cach noi suy*/
    for (i=0; i < ni; i++) {
        /*bottom - canh duoi (temp_bl = 200.0f, temp_br = 500.0f)*/
        j = 0;
        i2d = i + ni*j;
        h_temp1[i2d] = temp_bl + (temp_br-temp_bl)*(float)i/(float)(ni-1);

        /*top - canh tren (temp_tl = 200.0f, temp_tr = 500.0f)*/
        j = nj-1;
        i2d = i + ni*j;
        h_temp1[i2d] = temp_tl + (temp_tr-temp_tl)*(float)i/(float)(ni-1);
    }

    for (j=0; j < nj; j++) {
        /*left - canh trai (temp_bl = 200.0f, temp_tl = 200.0f)*/
        i = 0;
        i2d = i + ni*j;
        h_temp1[i2d] = temp_bl + (temp_tl-temp_bl)*(float)j/(float)(nj-1);

        /*right -  canh phai (temp_br = 500.0f, temp_tr = 500.0f)*/
        i = ni-1;
        i2d = i + ni*j;
	    h_temp1[i2d] = temp_br + (temp_tr-temp_br)*(float)j/(float)(nj-1);
    }

	/*sao chep du lieu tu mang h_temp1 sang h_temp2*/
	cudaMemcpy(h_temp2, h_temp1, sizeof(float)*ni*nj, cudaMemcpyHostToHost);
	      
	/*cap phat mang luu tru nhiet do tren thiet bi*/
    cudaMalloc((void **)&d_temp1, sizeof(float)*ni*nj);
    cudaMalloc((void **)&d_temp2, sizeof(float)*ni*nj);

	/*truyen du lieu nhiet do tu host toi thiet bi*/
    cudaMemcpy((void *)d_temp1, (void *)h_temp1, sizeof(float)*ni*nj, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_temp2, (void *)h_temp1, sizeof(float)*ni*nj, cudaMemcpyHostToDevice);
    
    
//    cudaEventCreate(&start);						/*do thoi gian bat dau su dung <cuda_runtime.h>*/
//	  cudaEventCreate(&stop);
//    cudaEventRecord(start, 0);

	startclock = clock();							/*do thoi gian su dung thu vien C: <time.h>*/

    for (istep=0; istep < nstep; istep++) {
        //printf("%i\n", istep);
		if (USE_CPU == 1) {                                           
            cpu_DiffusionEquation(ni, nj, dt, h_temp1, h_temp2);
	    
			// doi cho cac con tro cac mang nhiet do
            tmp_temp = h_temp1;
            h_temp1 = h_temp2;
            h_temp2 = tmp_temp;
            
        }
		if (USE_GPU == 1) {
	    
			// dinh nghia cau truc luoi va khoi
			dim3 Grid(Grid_Dim, Grid_Dim);			/*cau truc luoi*/
			dim3 Block(Block_Dim, Block_Dim);		/*cau truc khoi*/			           
            
			/*goi ham kernel tinh toan tren GPU*/ 
            gpu_DiffusionEquation<<<Grid, Block>>>(ni, nj, dt, d_temp1, d_temp2);
            /*doi cho cac con tro cac mang nhiet do*/						     
            tmp_temp = d_temp1;
            d_temp1 = d_temp2;
            d_temp2 = tmp_temp;
        }    
    } 

	stopclock = clock();
	elapsedtime = ((double)(stopclock-startclock))/CLOCKS_PER_SEC;
	printf("Thoi gian tinh toan: %f s.\n", elapsedtime);

//  cudaThreadSynchronize();
//	cudaEventRecord(stop, 0);									//do thoi gian ket thuc
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&elapsed_time, start, stop);
//  printf("\nThoi gian tinh toan: %f ms.\n", elapsed_time);

	/*sao chep mang ket qua tu GPU toi CPU*/
	if (USE_CPU == 0) {
        cudaMemcpy((void *)h_temp1, (void *)d_temp1, sizeof(float)*ni*nj, cudaMemcpyDeviceToHost);
    }
    
	/*ghi ket qua vao cac file cpu_output va gpu_out de so sanh*/

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
	
	/*giai phong bo nho tren GPU va CPU*/
	free(h_temp1);
	free(h_temp2);
	cudaFree(d_temp1);
	cudaFree(d_temp2);

//	cudaEventDestroy(start);
//	cudaEventDestroy(stop);

	return 0;
	getch();
}