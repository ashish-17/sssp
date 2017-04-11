#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

__global__ void warp_count_kernel(GraphEdge_t* edges, uint* warp_count, uint nEdges, uint* is_change) {

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int threadCount = blockDim.x * gridDim.x;
	int lane = threadId & (32 - 1);
	int countWarps = threadCount % 32 ? threadCount / 32 + 1 : threadCount / 32;
	int nEdgesPerWarp = nEdges % countWarps == 0 ? nEdges / countWarps : nEdges / countWarps + 1;
	int warpId = threadId / 32;
	int beg = nEdgesPerWarp*warpId;
	int end = beg + nEdgesPerWarp - 1;

	uint mask = 0;
	uint maskCount = 0;
	int predicate = 0;
	GraphEdge_t *edge;
	

	for (int i = beg + lane; i<= end && i< nEdges; i += 32) {
		edge = edges + i;

		predicate = is_change[edge->src] == 1;
		mask = __ballot(predicate);
		maskCount = __popc(mask);
		warp_count[warpId] += maskCount; 
	}
}

__global__ void filter_kernel(GraphEdge_t* edges, uint *edge_idx, uint* warp_offset, uint nEdges, uint* is_change) {

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int threadCount = blockDim.x * gridDim.x;
	int lane = threadId & (32 - 1);
	int countWarps = threadCount % 32 ? threadCount / 32 + 1 : threadCount / 32;
	int nEdgesPerWarp = nEdges % countWarps == 0 ? nEdges / countWarps : nEdges / countWarps + 1;
	int warpId = threadId / 32;
	int beg = nEdgesPerWarp*warpId;
	int end = beg + nEdgesPerWarp - 1;

	uint mask = 0;
	uint localId = 0;
	int predicate = 0;
	GraphEdge_t *edge;
	uint offset = 0;
	for (int i = beg + lane; i<= end && i< nEdges; i += 32) {
		edge = edges + i;

		predicate = is_change[edge->src] == 1;
		mask = __ballot(predicate);
		localId = __popc(mask << (32-lane));
		if (predicate == 1) {
			edge_idx[offset + localId + warp_offset[warpId]] = i;
		}

		offset += __popc(mask);
	}
}

__global__ void prefix_sum(uint* warp_count, uint count_warps, uint *nEdges) {

	__shared__ uint shared_mem[2048];

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int threadCount = blockDim.x * gridDim.x;
	if (threadId == 0) {
		*nEdges = warp_count[count_warps - 1];
	}
	
	if (threadId < count_warps)
		shared_mem[threadId] = warp_count[threadId];

	__syncthreads();

	for (int offset = 1; offset < count_warps; offset *= 2) {
		for (int idx = threadId; idx < count_warps; idx += threadCount) {
			if (idx >= offset) {
				warp_count[idx] += warp_count[idx - offset];
			}
		}

		__syncthreads();
	}	
	
	if (threadId == 0) {
		*nEdges += warp_count[count_warps - 1];
	}

	
	if (threadId < count_warps)
		warp_count[threadId] -= shared_mem[threadId];
}

__global__ void neighbourHandling_kernel(GraphEdge_t *edges, uint *edge_idx, uint nEdges, uint* d_curr, uint* d_prev, int *is_change, uint* is_src_changed) {

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
		edge = edges + edge_idx[i];
		src = edge->src;
		dest = edge->dest;
		weight = edge->weight;

		tmp = d_prev[src] + weight;
		if (tmp < d_prev[dest]) {
			atomicMin(&d_curr[dest], tmp);
			*is_change = 1;
			is_src_changed[dest] = 1;
		}
	} 
}

void neighbourHandler(GraphEdge_t* edges, uint nEdges, uint nVertices, uint* distance, int bsize, int bcount, int isIncore) {
	GraphEdge_t *d_edges;
	uint* d_edge_idx;
	uint* d_distances_curr;
	uint* d_distances_prev;
	uint* d_distances_dummy;
	uint* d_warp_count;
	uint* d_nEdges;
	uint* d_change;
	int *d_is_changed;
	int h_is_changed;
	uint count_to_process = nEdges;
	uint* h_edge_idx = new uint[nEdges];
	int count_iterations = 0;
	int count_warps = bsize * bcount % 32 == 0 ? bsize * bcount / 32 : (bsize * bcount / 32) + 1 ;

	cudaMalloc((void**)&d_edges, sizeof(GraphEdge_t)*nEdges);
	cudaMalloc((void**)&d_edge_idx, sizeof(uint)*nEdges);
	cudaMalloc((void**)&d_nEdges, sizeof(uint));
	cudaMalloc((void**)&d_distances_curr, sizeof(uint)*nVertices);
	cudaMalloc((void**)&d_distances_prev, sizeof(uint)*nVertices);
	cudaMalloc((void**)&d_distances_dummy, sizeof(uint)*nVertices);
	cudaMalloc((void**)&d_change, sizeof(uint)*nVertices);
	cudaMalloc((void**)&d_warp_count, sizeof(uint)*count_warps);
	cudaMalloc((void**)&d_is_changed, sizeof(int));

	cudaMemcpy(d_edges, edges, sizeof(GraphEdge_t)*nEdges, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nEdges, &nEdges, sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(d_distances_curr, distance, sizeof(uint)*nVertices, cudaMemcpyHostToDevice);
	cudaMemcpy(d_distances_prev, distance, sizeof(uint)*nVertices, cudaMemcpyHostToDevice);

	for (int i = 0; i < nEdges; ++i) {
		h_edge_idx[i] = i;
	}

	cudaMemcpy(d_edge_idx, h_edge_idx, sizeof(uint)*nEdges, cudaMemcpyHostToDevice);
	
	double filter_time = 0.0;
	double processing_time = 0.0;

	for (int i = 0; i < nVertices-1; ++i) {
		setTime();
	
		cudaMemset(d_is_changed, 0, sizeof(int));
		cudaMemset(d_change, 0, sizeof(uint)*nVertices);
		cudaMemset(d_warp_count, 0, sizeof(uint)*count_warps);
		cudaMemcpy(d_distances_dummy, d_distances_curr, sizeof(uint)*nVertices, cudaMemcpyDeviceToDevice);

		if (isIncore == 1)
			neighbourHandling_kernel<<<bcount, bsize>>>(d_edges, d_edge_idx, count_to_process, d_distances_curr, d_distances_curr, d_is_changed, d_change);
		else
			neighbourHandling_kernel<<<bcount, bsize>>>(d_edges, d_edge_idx, count_to_process, d_distances_curr, d_distances_prev, d_is_changed, d_change);
		
		cudaDeviceSynchronize();

		if (isIncore == 0)
			cudaMemcpy(d_distances_prev, d_distances_curr, sizeof(uint)*nVertices, cudaMemcpyDeviceToDevice);

		count_iterations++;

		cudaMemcpy(&h_is_changed, d_is_changed, sizeof(int), cudaMemcpyDeviceToHost);
		
		processing_time += getTime();
		
		if (h_is_changed == 0) {
			break;
		}

		setTime();

		warp_count_kernel<<<bcount, bsize>>>(d_edges, d_warp_count, nEdges, d_change);
		
		cudaDeviceSynchronize();

		prefix_sum<<<bcount, bsize>>>(d_warp_count, count_warps, d_nEdges);
		
		cudaDeviceSynchronize();

		filter_kernel<<<bcount, bsize>>>(d_edges, d_edge_idx, d_warp_count, nEdges, d_change);
		
		cudaDeviceSynchronize();
		
		cudaMemcpy(&count_to_process, d_nEdges, sizeof(uint), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_edge_idx, d_edge_idx, sizeof(uint)*nEdges, cudaMemcpyDeviceToHost);
		filter_time += getTime();
	}

	std::cout << "Took "<<count_iterations << " iterations " << processing_time + filter_time << "ms.(filter - "<<filter_time<<"ms processing - "<<processing_time<<"ms)\n";

	cudaMemcpy(distance, d_distances_curr, sizeof(uint)*nVertices, cudaMemcpyDeviceToHost);

	delete[] h_edge_idx;
	cudaFree(d_edges);
	cudaFree(d_edge_idx);
	cudaFree(d_nEdges);
	cudaFree(d_change);
	cudaFree(d_distances_curr);
	cudaFree(d_distances_prev);
	cudaFree(d_distances_dummy);
	cudaFree(d_warp_count);
	cudaFree(d_is_changed);
}
