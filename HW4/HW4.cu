// Last update: 16/12/2020
#include <stdio.h>
#include <stdint.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}\




struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// Sequential Radix Sort
void sortByHost(const uint32_t * in, int n,
                uint32_t * out)
{
    int * bits = (int *)malloc(n * sizeof(int));
    int * nOnesBefore = (int *)malloc(n * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    // Loop from LSB (Least Significant Bit) to MSB (Most Significant Bit)
	// In each loop, sort elements according to the current bit from src to dst 
	// (using STABLE counting sort)
    for (int bitIdx = 0; bitIdx < sizeof(uint32_t) * 8; bitIdx++)
    {
        // Extract bits
        for (int i = 0; i < n; i++)
            bits[i] = (src[i] >> bitIdx) & 1;

        // Compute nOnesBefore
        nOnesBefore[0] = 0;
        for (int i = 1; i < n; i++)
            nOnesBefore[i] = nOnesBefore[i-1] + bits[i-1];

        // Compute rank and write to dst
        int nZeros = n - nOnesBefore[n-1] - bits[n-1];
        for (int i = 0; i < n; i++)
        {
            int rank;
            if (bits[i] == 0)
                rank = i - nOnesBefore[i];
            else
                rank = nZeros + nOnesBefore[i];
            dst[rank] = src[i];
        }

        // Swap src and dst
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }

    // Does out array contain results?
    memcpy(out, src, n * sizeof(uint32_t));

    // Free memory
    free(originalSrc);
    free(bits);
    free(nOnesBefore);
}
__device__ int bCount = 0;
volatile __device__ int bCount1 = 0;
// song song hóa bước extract bit
__global__ void extractBitKernel(const uint32_t* in, uint32_t* out, int n, int bitIdx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <n)
    {
        out[i] = int((in[i] >> bitIdx) & 1);
    }
    __syncthreads();
}

// song song hóa bước scan bit
__global__ void scanBitKernel(const uint32_t* in, uint32_t* out, volatile uint32_t* bSums, int n)
{
    
    extern __shared__ int s_data[];
    __shared__ int bi;
    
    if ( threadIdx.x == 0)
    {
        bi = atomicAdd(&bCount, 1);
    }
    __syncthreads();
    //copy data from GMEM to SMEM
    int i1, i2;
    i1 = bi * 2 * blockDim.x + threadIdx.x;
    i2 = i1 + blockDim.x;

    if (i1 >= n && i2 >= n)
        return;
        s_data[threadIdx.x] = (i1 <= 0) ?0 : in[i1 - 1];
        s_data[threadIdx.x + blockDim.x] = (i2 <= 0) ?0 : in[i2 - 1];

	__syncthreads();

    // scan
    for(int stride = 1; stride < 2* blockDim.x; stride*= 2)
    {
        int s_dataIdx = (threadIdx.x+ 1)*2* stride - 1;
        if( s_dataIdx < 2* blockDim.x)
        {
            s_data[s_dataIdx] += s_data[s_dataIdx - stride];
        }
        __syncthreads();
    }

    for(int stride = blockDim.x/2; stride > 0; stride /= 2)
    {
        int s_dataIdx = (threadIdx.x + 1)*2*stride - 1+ stride;
        if(s_dataIdx < 2* blockDim.x)
        {
            s_data[s_dataIdx] += s_data[s_dataIdx - stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0)
    {
        bSums[bi] = s_data[2*blockDim.x - 1];

        if (bi > 0)
        {
            while (bCount1 < bi) {}
            bSums[bi] += bSums[bi - 1];
            __threadfence();
        }
        bCount1 += 1;
    }
    __syncthreads();
    // copy data from SMEM to GMEM
    if (i1 < n)
        out[i1] = s_data[threadIdx.x];

    if (i2 < n)
        out[i2] = s_data[threadIdx.x + blockDim.x];
    
    if (bi > 0)
    {
        if (i1 < n) out[i1] += bSums[bi - 1];
        if (i2 < n) out[i2] += bSums[bi - 1];
    }
}



__global__ void computeRankKernel(const uint32_t * in, uint32_t* out, const uint32_t* bits, uint32_t* nOnesBefore, int n,  int bitIdx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nZeros = n - nOnesBefore[n-1] - bits[n - 1];

    int rank;
    rank = (bits[i] == 0) ? i - nOnesBefore[i] : nZeros + nOnesBefore[i];
    out[rank] = in[i];
}
// Parallel Radix Sort
void sortByDevice(const uint32_t * in, int n, uint32_t * out, int blockSize)
{
    // TODO
    
    int blkDataSize = 2* blockSize;
    int blkgridSize = (n - 1) / blkDataSize + 1;
    int gridSize = (n - 1) / blockSize +1;
    size_t nBytes = n*sizeof(uint32_t);
    size_t sMem = blkDataSize* sizeof(int);
    uint32_t *d_in, *d_out, *d_bits, *d_blkSums, *d_nOnesBefore;

    CHECK(cudaMalloc(&d_in, nBytes));
    CHECK(cudaMalloc(&d_out, nBytes));
    CHECK(cudaMalloc(&d_bits, nBytes));
    CHECK(cudaMalloc(&d_nOnesBefore, nBytes));
    CHECK(cudaMalloc(&d_blkSums, blkgridSize * sizeof(int)));

    CHECK(cudaMemcpy(d_in, in, nBytes, cudaMemcpyHostToDevice));  

    // Loop from LSB (Least Significant Bit) to MSB (Most Significant Bit)
	// In each loop, sort elements according to the current bit from src to dst 
	// (using STABLE counting sort)
    for (int bitIdx = 0; bitIdx< sizeof(uint32_t) * 8; bitIdx ++)
    {
        // reset bCount, bCuont1
        int zero = 0; 
        CHECK(cudaMemcpyToSymbol(bCount,&zero, sizeof(int)));
        CHECK(cudaMemcpyToSymbol(bCount1,&zero, sizeof(int)));

        //extract bit
        extractBitKernel<<<gridSize, blockSize>>>(d_in, d_bits, n, bitIdx);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());
        
        //scan bit
        scanBitKernel<<<blkgridSize, blockSize, sMem>>>(d_bits, d_nOnesBefore, d_blkSums, n);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        // Compute rank and write to dst
        computeRankKernel<<<gridSize, blockSize>>>(d_in, d_out,  d_bits, d_nOnesBefore, n, bitIdx);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        //swap d_in and d_out
        uint32_t *tmp;
        tmp = d_in;
        d_in = d_out;
        d_out = tmp;
    }

    CHECK(cudaMemcpy(out, d_out, nBytes, cudaMemcpyDeviceToHost));

    //free memmory
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_bits);
    cudaFree(d_blkSums);
    cudaFree(d_nOnesBefore);

} 


// Radix Sort
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        bool useDevice=false, int blockSize=1)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
    	printf("\nRadix Sort by host\n");
        sortByHost(in, n, out);
    }
    else // use device
    {
    	printf("\nRadix Sort by device\n");
        sortByDevice(in, n, out, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

void printArray(uint32_t * a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    //int n = 50; // For test by eye
    int n = (1 << 24) + 1;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
    {
        //in[i] = rand() % 255; // For test by eye
        in[i] = rand();
    }
    //printArray(in, n); // For test by eye

    // DETERMINE BLOCK SIZE
    int blockSize = 512; // Default 
    if (argc == 2)
        blockSize = atoi(argv[1]);

    // SORT BY HOST
    sort(in, n, correctOut);
    //printArray(correctOut, n); // For test by eye
    
    // SORT BY DEVICE
    sort(in, n, out, true, blockSize);
    //printArray(out, n); // For test by eye
    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
