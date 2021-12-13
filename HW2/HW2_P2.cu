#include <stdio.h>

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() {
    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);
  }

  void Stop() { cudaEventRecord(stop, 0); }

  float Elapsed() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

__global__ void addVecKernel(int *in1, int *in2, int n, int *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    out[i] = in1[i] + in2[i];
  }
}

void addVec(int *in1, int *in2, int n, int *out, bool useDevice = false,
            dim3 blockSize = dim3(1), int nStreams = 1) {
  if (useDevice == false) {
    for (int i = 0; i < n; i++) {
      out[i] = in1[i] + in2[i];
    }
  } else // Use device
  {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printf("GPU name: %s\n", devProp.name);
    printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

    // Pin host memory regions (allocated by malloc)
    // so that we can use cudaMemcpyAsync
    size_t nBytes = n * sizeof(int);
    CHECK(cudaHostRegister(in1, nBytes, cudaHostRegisterDefault));
    CHECK(cudaHostRegister(in2, nBytes, cudaHostRegisterDefault));
    CHECK(cudaHostRegister(out, nBytes, cudaHostRegisterDefault));

    // TODO: Allocate device memory regions
    int *d_in1, *d_in2, *d_out;
    CHECK(cudaMalloc(&d_in1, nBytes));
    CHECK(cudaMalloc(&d_in2, nBytes));
    CHECK(cudaMalloc(&d_out, nBytes));
    // TODO: Create "nStreams" device streams
    cudaStream_t *streams =
        (cudaStream_t *)malloc(nStreams * sizeof(cudaStream_t));
    for (int i = 0; i < nStreams; i++)
      CHECK(cudaStreamCreate(&streams[i]));
    GpuTimer timer;
    timer.Start();

    // TODO: Send jobs (H2D, kernel, D2H) to device streams
    int streamSize = n / nStreams; // size of each stream
    dim3 gridSize = ((n - 1) / blockSize.x + 1); 
    int streamBytes = streamSize * sizeof(int); // number of byte to copy data in function cudaMemcpyAsync()

    for (int i = 0; i < nStreams; i++) {
      int offset = streamSize * i;
      if ((i == nStreams - 1) && (nStreams != 1)) {
        streamBytes += (n % nStreams) * sizeof(int);
        streamSize += n % nStreams;
      }

      // copy data from host to device
      CHECK(cudaMemcpyAsync(&d_in1[offset], &in1[offset], streamBytes,
                            cudaMemcpyHostToDevice, streams[i]));
      CHECK(cudaMemcpyAsync(&d_in2[offset], &in2[offset], streamBytes,
                            cudaMemcpyHostToDevice, streams[i]));

      addVecKernel<<<gridSize, blockSize, 0, streams[i]>>>(
          &d_in1[offset], &d_in2[offset], streamSize, &d_out[offset]);
      // copy data form device to host
      CHECK(cudaMemcpyAsync(&out[offset], &d_out[offset], streamBytes,
                            cudaMemcpyDeviceToHost, streams[i]));
    }
    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time of all device streams: %f ms\n\n", time);

    // TODO: Destroy device streams
    for (int i = 0; i < nStreams; i++)
      CHECK(cudaStreamDestroy(streams[i]));
    free(streams);
    // TODO: Free device memory regions
    CHECK(cudaFree(d_in1));
    CHECK(cudaFree(d_in2));
    CHECK(cudaFree(d_out));
    // Unpin host memory regions
    CHECK(cudaHostUnregister(in1));
    CHECK(cudaHostUnregister(in2));
    CHECK(cudaHostUnregister(out));
  }
}

int main(int argc, char **argv) {
  int n;
  int *in1, *in2;
  int *out, *correctOut;

  // Input data into n
  n = (1 << 24) + 1;
  printf("n =  %d\n\n", n);

  // Allocate memories for in1, in2, out
  size_t nBytes = n * sizeof(int);
  in1 = (int *)malloc(nBytes);
  in2 = (int *)malloc(nBytes);
  out = (int *)malloc(nBytes);
  /*
  CHECK(cudaMallocHost(&in1, nBytes));
  CHECK(cudaMallocHost(&in2, nBytes));
  CHECK(cudaMallocHost(&out, nBytes));
  */
  correctOut = (int *)malloc(nBytes);

  // Input data into in1, in2
  for (int i = 0; i < n; i++) {
    in1[i] = rand() & 0xff; // Random int in [0, 255]
    in2[i] = rand() & 0xff; // Random int in [0, 255]
  }

  // Add in1 & in2 on host
  addVec(in1, in2, n, correctOut);

  // Add in1 & in2 on device
  dim3 blockSize(512); // Default
  int nStreams = 1;    // Default
  if (argc >= 2) {
    blockSize.x = atoi(argv[1]);
    if (argc >= 3) {
      nStreams = atoi(argv[2]);
    }
  }
  addVec(in1, in2, n, out, true, blockSize, nStreams);

  // Check correctness
  for (int i = 0; i < n; i++) {
    if (out[i] != correctOut[i]) {
      printf("INCORRECT :(\n");
      return 1;
    }
  }
  printf("CORRECT :)\n");

  free(in1);
  free(in2);
  free(out);
  /*
  CHECK(cudaFreeHost(in1));
  CHECK(cudaFreeHost(in2));
  CHECK(cudaFreeHost(out));
  */
  free(correctOut);
}
