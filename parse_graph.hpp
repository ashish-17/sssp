#ifndef PARSE_GRAPH_HPP
#define PARSE_GRAPH_HPP

#include <fstream>

#include "initial_graph.hpp"

namespace parse_graph {
	uint parse(
		std::ifstream& inFile,
		std::vector<initial_vertex>& initGraph,
		const long long arbparam,
		const bool nondirected );

	void covertToGraphEdgeFormat(std::vector<initial_vertex>& graph, GraphEdge_t* edges);
	void updateDistances(std::vector<initial_vertex>& graph, uint*d);
	void writeOutput(std::vector<initial_vertex> &graph, char* outputFileName);
}

#endif	//	PARSE_GRAPH_HPP
