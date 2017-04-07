#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

#include "opt.cu"
#include "impl2.cu"
#include "impl1.cu"

enum class ProcessingType {Push, Neighbor, Own, Unknown};
enum SyncMode {InCore, OutOfCore};
enum SyncMode syncMethod;
enum SmemMode {UseSmem, UseNoSmem};
enum SmemMode smemMethod;
enum EdgeSortOrder {SortBySrc, SortByDest};
enum EdgeSortOrder sortOrder;

static inline int graphEdgeComparatorSrc(void* e1, void*e2) {
	GraphEdge_t *e11 = (GraphEdge_t*)e1;
	GraphEdge_t *e22 = (GraphEdge_t*)e2;
	if (e11->src != e22->src) { 
		return e11->src - e22->src;
	} else {
		return e11->dest - e22->dest;
	}
}

static inline int graphEdgeComparatorDest(void* e1, void*e2) {
	GraphEdge_t *e11 = (GraphEdge_t*)e1;
	GraphEdge_t *e22 = (GraphEdge_t*)e2;
	if (e11->dest != e22->dest) { 
		return e11->dest - e22->dest;
	} else {
		return e11->src - e22->src;
	}
}

void testCorrectness(std::vector<initial_vertex> * parsedGraph, char* outputFileName);

// Execution entry point.
int main( int argc, char** argv )
{

	std::string usage =
		"\tRequired command line arguments:\n\
		Input file: E.g., --input in.txt\n\
		Block size: E.g., --bsize 512\n\
		Block count: E.g., --bcount 192\n\
		Output path: E.g., --output output.txt\n\
		Processing method: E.g., --method bmf (bellman-ford), or tpe (to-process-edge), or opt (one further optimizations)\n\
		Shared memory usage: E.g., --usesmem yes, or no \n\
		Sync method: E.g., --sync incore, or outcore\n\
		Sort Order: E.g., --sort src, or dest\n";

	try {

		std::ifstream inputFile;
		std::ofstream outputFile;
		char outputFileName[256]; 
		int selectedDevice = 0;
		int bsize = 0, bcount = 0;
		int vwsize = 32;
		int threads = 1;
		long long arbparam = 0;
		bool nonDirectedGraph = false;		// By default, the graph is directed.
		ProcessingType processingMethod = ProcessingType::Unknown;
		syncMethod = OutOfCore;
		sortOrder = SortBySrc;


		/********************************
		 * GETTING INPUT PARAMETERS.
		 ********************************/

		for( int iii = 1; iii < argc; ++iii )
			if ( !strcmp(argv[iii], "--method") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "bmf") )
					processingMethod = ProcessingType::Push;
				else if ( !strcmp(argv[iii+1], "tpe") )
					processingMethod = ProcessingType::Neighbor;
				else if ( !strcmp(argv[iii+1], "opt") )
					processingMethod = ProcessingType::Own;
				else{
					std::cerr << "\n Un-recognized method parameter value \n\n";
					exit;
				}   
			}
			else if ( !strcmp(argv[iii], "--sync") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "incore") )
					syncMethod = InCore;
				else if ( !strcmp(argv[iii+1], "outcore") )
					syncMethod = OutOfCore;
				else {
					std::cerr << "\n Un-recognized sync parameter value \n\n";
					exit;
				}  

			}
			else if ( !strcmp(argv[iii], "--sort") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "dest") ) {
					sortOrder = SortByDest;
				}
				else {
					sortOrder = SortBySrc;
				}
			}
			else if ( !strcmp(argv[iii], "--usesmem") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "yes") )
					smemMethod = UseSmem;
				if ( !strcmp(argv[iii+1], "no") )
					smemMethod = UseNoSmem;
				else{
					std::cerr << "\n Un-recognized usesmem parameter value \n\n";
					exit;
				}  
			}
			else if( !strcmp( argv[iii], "--input" ) && iii != argc-1 /*is not the last one*/)
				openFileToAccess< std::ifstream >( inputFile, std::string( argv[iii+1] ) );
			else if( !strcmp( argv[iii], "--output" ) && iii != argc-1 /*is not the last one*/) {
				openFileToAccess< std::ofstream >( outputFile, std::string( argv[iii+1] ) );
				strcpy(outputFileName, argv[iii+1]);
			}
			else if( !strcmp( argv[iii], "--bsize" ) && iii != argc-1 /*is not the last one*/)
				bsize = std::atoi( argv[iii+1] );
			else if( !strcmp( argv[iii], "--bcount" ) && iii != argc-1 /*is not the last one*/)
				bcount = std::atoi( argv[iii+1] );

		if(bsize <= 0 || bcount <= 0){
			std::cerr << "Usage: " << usage;
			exit;
			throw std::runtime_error("\nAn initialization error happened.\nExiting.");
		}
		if( !inputFile.is_open() || processingMethod == ProcessingType::Unknown ) {
			std::cerr << "Usage: " << usage;
			throw std::runtime_error( "\nAn initialization error happened.\nExiting." );
		}
		if( !outputFile.is_open() ) {
			openFileToAccess< std::ofstream >( outputFile, "out.txt" );
			strcpy(outputFileName, "out.txt");
		}
		CUDAErrorCheck( cudaSetDevice( selectedDevice ) );
		std::cout << "Device with ID " << selectedDevice << " is selected to process the graph.\n";


		/********************************
		 * Read the input graph file.
		 ********************************/

		std::cout << "Collecting the input graph ...\n";
		std::vector<initial_vertex> parsedGraph( 0 );
		uint nEdges = parse_graph::parse(
				inputFile,		// Input file.
				parsedGraph,	// The parsed graph.
				arbparam,
				nonDirectedGraph );		// Arbitrary user-provided parameter.
		std::cout << "Input graph collected with " << parsedGraph.size() << " vertices and " << nEdges << " edges.\n";


		/********************************
		 * Process the graph.
		 ********************************/
		unsigned int vertex_size = parsedGraph.size();
		unsigned int *distance = new unsigned int[vertex_size];
		GraphEdge_t *edges = new GraphEdge_t[nEdges];
		for (unsigned int i = 0; i < vertex_size; ++i)
			distance[i] = INFINITY;
		distance[0]=0;
		
		parse_graph::covertToGraphEdgeFormat(parsedGraph, edges);
		if (sortOrder == SortBySrc) {
			mergeSortSeq(edges, sizeof(GraphEdge_t), nEdges, graphEdgeComparatorSrc);	
		} else {
			mergeSortSeq(edges, sizeof(GraphEdge_t), nEdges, graphEdgeComparatorDest);	
		}
		
		//for (int x = 0; x < nEdges; ++x) {std::cout<<"\nEdge - "<<x<<" "<<edges[x].src<<" - "<<edges[x].dest<<" - "<<edges[x].weight<<"\n";}
 
		switch(processingMethod){
			case ProcessingType::Push:
				puller(edges, nEdges, distance, bsize, bcount, syncMethod == InCore);
				break;
			case ProcessingType::Neighbor:
				neighborHandler(&parsedGraph, bsize, bcount);
				break;
			default:
				own(&parsedGraph, bsize, bcount);
		}

		/********************************
		 * It's done here.
		 ********************************/
		
		parse_graph::writeOutput(parsedGraph, outputFileName);
		testCorrectness(&parsedGraph, outputFileName);
		CUDAErrorCheck( cudaDeviceReset() );
		std::cout << "Done.\n";

		delete[] distance;
		delete[] edges;
		return( EXIT_SUCCESS );

	}
	catch( const std::exception& strException ) {
		std::cerr << strException.what() << "\n";
		return( EXIT_FAILURE );
	}
	catch(...) {
		std::cerr << "An exception has occurred." << std::endl;
		return( EXIT_FAILURE );
	}

}



void testCorrectness(std::vector<initial_vertex> * parsedGraph, char* outputFileName) {
	std::cout << std::endl << "TESTING CORRECTNESS" << std::endl;

	std::cout << "RUNNING SEQUENTIAL BMF..." << std::endl;
	unsigned int vertex_size = (*parsedGraph).size();
	unsigned int *d= new unsigned int[vertex_size];
	for (unsigned int i = 0; i < vertex_size; ++i)
		d[i] = INFINITY;
	d[0]=0;

	int change = 0;
	for (unsigned int k = 1; k < vertex_size; k++){
		for (unsigned int i = 0; i < vertex_size; i++){
			std::vector<neighbor> nbrs = (*parsedGraph)[i].nbrs;
			for (unsigned int j = 0; j < nbrs.size(); ++j){
				unsigned int u = nbrs[j].srcIndex;
				unsigned int v = i;
				unsigned int w = nbrs[j].edgeValue.weight;
				if (d[u] == INFINITY)
					continue;

				if ((d[u] + w) < d[v]){
					d[v] = d[u]+w;
					change = 1;
				}
			}
		}
		if (change == 0)
			break;
		change = 0;
	}

	//Compare the distance array and the parallel output file
	std::ifstream outputFile;
	openFileToAccess< std::ifstream >( outputFile, std::string( outputFileName ) );

	std::string line;
	unsigned int i = 0;
	unsigned int incorrect = 0;
	while (getline(outputFile,line)) {
		std::string curr = (d[i] < INFINITY) ? (std::to_string(i) + ": " + std::to_string(d[i])):(std::to_string(i) +": " + "INF");

		// std::cout << std::to_string(line.compare(curr)) << std::endl;

		if(line.compare(curr) != 0) {
			incorrect++;
			std::cout << "Correct: " << curr << "\tYours: " << line << std::endl;
		}
		i++;
	}
	if(i != vertex_size) {
		std::cout << "Insufficient vertices found in outputfile" << std::endl;
		std::cout << "Expected: " << vertex_size << "Found: " << i << std::endl;
		return;
	}
	std::cout << "Correct: " << std::to_string(vertex_size-incorrect) << "\t Incorrect: " << std::to_string(incorrect) << " \t Total: " << std::to_string(vertex_size) << std::endl;
	outputFile.close();
	delete[] d;
}
