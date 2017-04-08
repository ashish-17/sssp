#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

__global__ void pulling_kernel(GraphEdge_t* edges, uint nEdges, uint* d_curr, uint* d_prev, int* is_changed) {

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int threadCount = blockDim.x * gridDim.x;
	int lane = threadId & (32 - 1);
	int countWarps = threadCount % 32 ? threadCount / 32 + 1 : threadCount / 32;
	int nEdgesPerWarp = nEdges % countWarps == 0 ? nEdges / countWarps : nEdges / countWarps + 1;
	int warpId = threadId / 32;
	int beg = nEdgesPerWarp*warpId;
	int end = beg + nEdgesPerWarp - 1;

	uint src = 0;
	uint dest = 0;
	uint weight = 0;
	uint tmp = 0;
	GraphEdge_t *edge;
	for (int i = beg + lane; i<= end && i< nEdges; i += 32) {
		edge = edges + i;
		src = edge->src;
		dest = edge->dest;
		weight = edge->weight;

		tmp = d_prev[src] + weight;
		if (tmp < d_prev[dest]) {
			atomicMin(&d_curr[dest], tmp);
			*is_changed = 1;
		}
	} 
}

__device__ uint min_dist(uint val1, uint val2) {
	if (val1 < val2) {
		return val1;
	} else {
		return val2;
	}
}

__global__ void pulling_kernel_smem(GraphEdge_t* edges, uint nEdges, uint nVertices, uint* d_curr, uint* d_prev, int* is_changed) {

	__shared__ uint shared_mem[2048];

	const int nEdgesPerWarp = 64;
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int threadCount = blockDim.x * gridDim.x;
	int lane = threadId & (32 - 1);
	int countWarps = threadCount % 32 ? threadCount / 32 + 1 : threadCount / 32;
	int nEdgesPerIter = countWarps * nEdgesPerWarp;
	int nIters = nEdges % nEdgesPerIter ? nEdges / nEdgesPerIter + 1 : nEdges / nEdgesPerIter;

	uint src = 0;
	uint dest = 0;
	uint weight = 0;
	GraphEdge_t *edge;
	for (int iter = 0; iter < nIters; ++iter) {
		int warpId = threadId / 32 + iter*countWarps;
		int beg = warpId * nEdgesPerWarp;
		int end = beg + nEdgesPerWarp -1;

		for (int i = beg + lane; i<= end && i< nEdges; i += 32) {
			edge = edges + i;
			src = edge->src;
			dest = edge->dest;
			weight = edge->weight;

			shared_mem[threadIdx.x] = min_dist(d_prev[src] + weight, d_prev[dest]);

			__syncthreads();

			if ((lane >= 1) && (edges[i].dest == edges[i-1].dest)) {
				shared_mem[threadIdx.x] = min_dist(shared_mem[threadIdx.x], shared_mem[threadIdx.x-1]);
			}
			if (lane >= 2 && edges[i].dest == edges[i-2].dest) {
				shared_mem[threadIdx.x] = min_dist(shared_mem[threadIdx.x], shared_mem[threadIdx.x-2]);
			}
			if (lane >= 4 && edges[i].dest == edges[i-4].dest) {
				shared_mem[threadIdx.x] = min_dist(shared_mem[threadIdx.x], shared_mem[threadIdx.x-4]);
			}
			if (lane >= 8 && edges[i].dest == edges[i-8].dest) {
				shared_mem[threadIdx.x] = min_dist(shared_mem[threadIdx.x], shared_mem[threadIdx.x-8]);
			}
			if (lane >= 16 && edges[i].dest == edges[i-16].dest) {
				shared_mem[threadIdx.x] = min_dist(shared_mem[threadIdx.x], shared_mem[threadIdx.x-16]);
			}

			__syncthreads();

			if (((i < nEdges - 1) && edges[i].dest != edges[i+1].dest) || (i % 32 == 31 || i == nEdges - 1)) {
				if (d_curr[dest] > shared_mem[threadIdx.x])
					atomicMin(&d_curr[dest], shared_mem[threadIdx.x]);
			}
		}
	}

	__syncthreads();

	int nVerticesPerWarp = nVertices % countWarps == 0 ? nVertices / countWarps : nVertices / countWarps + 1;
	int warpId = threadId / 32;
	int beg = nVerticesPerWarp*warpId;
	int end = beg + nVerticesPerWarp - 1;
	for (int i = beg + lane; i<= end && i< nVertices && *is_changed == 0; i += 32) {
		if (d_prev[i] != d_curr[i]) {
			*is_changed = 1;
		}
	}
}

void puller(GraphEdge_t* edges, uint nEdges, uint nVertices, uint* distance, int bsize, int bcount, int isIncore, int useSharedMem) {
	GraphEdge_t *d_edges;
	uint* d_distances_curr;
	uint* d_distances_prev;
	int *d_is_changed;
	int h_is_changed;
	int count_iterations = 0;

	cudaMalloc((void**)&d_edges, sizeof(GraphEdge_t)*nEdges);
	cudaMalloc((void**)&d_distances_curr, sizeof(uint)*nVertices);
	cudaMalloc((void**)&d_distances_prev, sizeof(uint)*nVertices);
	cudaMalloc((void**)&d_is_changed, sizeof(int));

	cudaMemcpy(d_edges, edges, sizeof(GraphEdge_t)*nEdges, cudaMemcpyHostToDevice);
	cudaMemcpy(d_distances_curr, distance, sizeof(uint)*nVertices, cudaMemcpyHostToDevice);
	cudaMemcpy(d_distances_prev, distance, sizeof(uint)*nVertices, cudaMemcpyHostToDevice);

	setTime();

	for (int i = 0; i < nVertices-1; ++i) {
		cudaMemset(d_is_changed, 0, sizeof(int));

		if (isIncore == 1)
			pulling_kernel<<<bcount, bsize>>>(d_edges, nEdges, d_distances_curr, d_distances_curr, d_is_changed);
		else if (useSharedMem == 0)
			pulling_kernel<<<bcount, bsize>>>(d_edges, nEdges, d_distances_curr, d_distances_prev, d_is_changed);
		else
			pulling_kernel_smem<<<bcount, bsize>>>(d_edges, nEdges, nVertices, d_distances_curr, d_distances_prev, d_is_changed);

		cudaDeviceSynchronize();
		if (isIncore == 0)
			cudaMemcpy(d_distances_prev, d_distances_curr, sizeof(uint)*nVertices, cudaMemcpyDeviceToDevice);

		count_iterations++;

		cudaMemcpy(&h_is_changed, d_is_changed, sizeof(int), cudaMemcpyDeviceToHost);
		if (h_is_changed == 0) {
			break;
		}

	}

	std::cout << "Took "<<count_iterations << " iterations " << getTime() << "ms.\n";

	cudaMemcpy(distance, d_distances_curr, sizeof(uint)*nVertices, cudaMemcpyDeviceToHost);

	cudaFree(d_edges);
	cudaFree(d_distances_curr);
	cudaFree(d_distances_prev);
	cudaFree(d_is_changed);
}
