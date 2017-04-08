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

	GraphEdge_t *edge;
	for (int i = beg + lane; i<= end && i< nEdges; i += 32) {
		edge = edges + i;
		src = edge->src;
		dest = edge->dest;
		weight = edges->weight;
	
		if (d_prev[src] + weight < d_prev[dest]) {
			atomicMin(&d_curr[dest], d_prev[src] + weight);
			*is_changed = 1;
		}
	} 
}

void puller(GraphEdge_t* edges, uint nEdges, uint nVertices, uint* distance, int bsize, int bcount, int isIncore) {
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
		
		if (isIncore == 0)
			pulling_kernel<<<bcount, bsize>>>(d_edges, nEdges, d_distances_curr, d_distances_prev, d_is_changed);
		else
			pulling_kernel<<<bcount, bsize>>>(d_edges, nEdges, d_distances_curr, d_distances_curr, d_is_changed);

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
