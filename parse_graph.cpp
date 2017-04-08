#include <string>
#include <cstring>
#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include "parse_graph.hpp"
#include "utils.h"

uint parse_graph::parse(
		std::ifstream& inFile,
		std::vector<initial_vertex>& initGraph,
		const long long arbparam,
		const bool nondirected ) {

	const bool firstColumnSourceIndex = true;

	std::string line;
	char delim[3] = " \t";	//In most benchmarks, the delimiter is usually the space character or the tab character.
	char* pch;
	uint nEdges = 0;

	unsigned int Additionalargc=0;
	char* Additionalargv[ 61 ];

	// Read the input graph line-by-line.
	while( std::getline( inFile, line ) ) {
		if( line[0] < '0' || line[0] > '9' )	// Skipping any line blank or starting with a character rather than a number.
			continue;
		char cstrLine[256];
		std::strcpy( cstrLine, line.c_str() );
		uint firstIndex, secondIndex;

		pch = strtok(cstrLine, delim);
		if( pch != NULL )
			firstIndex = atoi( pch );
		else
			continue;
		pch = strtok( NULL, delim );
		if( pch != NULL )
			secondIndex = atoi( pch );
		else
			continue;

		uint theMax = std::max( firstIndex, secondIndex );
		uint srcVertexIndex = firstColumnSourceIndex ? firstIndex : secondIndex;
		uint dstVertexIndex = firstColumnSourceIndex ? secondIndex : firstIndex;
		if( initGraph.size() <= theMax )
			initGraph.resize(theMax+1);
		{ //This is just a block
			// Add the neighbor. A neighbor wraps edges
			neighbor nbrToAdd;
			nbrToAdd.srcIndex = srcVertexIndex;

			Additionalargc=0;
			Additionalargv[ Additionalargc ] = strtok( NULL, delim );
			while( Additionalargv[ Additionalargc ] != NULL ){
				Additionalargc++;
				Additionalargv[ Additionalargc ] = strtok( NULL, delim );
			}
			initGraph.at(srcVertexIndex).vertexValue.distance = ( srcVertexIndex != arbparam ) ? SSSP_INF : 0;
			initGraph.at(dstVertexIndex).vertexValue.distance = ( dstVertexIndex != arbparam ) ? SSSP_INF : 0;
			nbrToAdd.edgeValue.weight = ( Additionalargc > 0 ) ? atoi(Additionalargv[0]) : 1;

			initGraph.at(dstVertexIndex).nbrs.push_back( nbrToAdd );
			nEdges++;
		}
		if( nondirected ) {
			// Add the edge going the other way
			uint tmp = srcVertexIndex;
			srcVertexIndex = dstVertexIndex;
			dstVertexIndex = tmp;
			//swap src and dest and add as before

			neighbor nbrToAdd;
			nbrToAdd.srcIndex = srcVertexIndex;

			Additionalargc=0;
			Additionalargv[ Additionalargc ] = strtok( NULL, delim );
			while( Additionalargv[ Additionalargc ] != NULL ){
				Additionalargc++;
				Additionalargv[ Additionalargc ] = strtok( NULL, delim );
			}
			initGraph.at(srcVertexIndex).vertexValue.distance = ( srcVertexIndex != arbparam ) ? SSSP_INF : 0;
			initGraph.at(dstVertexIndex).vertexValue.distance = ( dstVertexIndex != arbparam ) ? SSSP_INF : 0;
			nbrToAdd.edgeValue.weight = ( Additionalargc > 0 ) ? atoi(Additionalargv[0]) : 1;
			initGraph.at(dstVertexIndex).nbrs.push_back( nbrToAdd );
			nEdges++;
		}
	}

	return nEdges;

}


void parse_graph::covertToGraphEdgeFormat(std::vector<initial_vertex>& graph, GraphEdge_t* edges) {

	int vertex_count = graph.size();
	int e = 0;
	for (int i = 0; i < vertex_count; ++i) {
		std::vector<neighbor> nbrs = graph[i].nbrs;
		for (int j = 0; j < nbrs.size(); ++j) {
			GraphEdge_t edge;
			edges[e].src = nbrs[j].srcIndex;
			edges[e].dest = i;
			edges[e].weight = nbrs[j].edgeValue.weight;
			e++;
		}
	}

}

void parse_graph::updateDistances(std::vector<initial_vertex>& graph, uint*d) {

	int vertex_count = graph.size();
	for (int i = 0; i < vertex_count; ++i) {
		graph.at(i).vertexValue.distance = d[i];
	}
}


void parse_graph::writeOutput(std::vector<initial_vertex> &graph, char* outputFileName) {
	std::cout << std::endl << "WRITTING O/P" << std::endl;
	std::ofstream outputFile;
	openFileToAccess< std::ofstream >(outputFile, std::string(outputFileName));
	int vertex_count = graph.size();
	char buffer[1024];
	for (int i = 0; i < vertex_count; ++i) {
		unsigned int d = graph.at(i).vertexValue.distance;
		memset(buffer, 0, 1024);
		sprintf(buffer, "%d: %s\n", i, (d == D_INFINITY) ? "INF": std::to_string(d).c_str());
		outputFile<<buffer;
	}
	

	outputFile.close();	
}
