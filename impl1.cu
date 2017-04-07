#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

__global__ void pulling_kernel(std::vector<initial_vertex> * peeps, int offset, int * anyChange){

    //update me based on my neighbors. Toggle anyChange as needed.
    //offset will tell you who I am.
}

void puller(GraphEdge_t* edges, uint nEdges, uint* distance, int bsize, int bcount, int isIncore) {
    setTime();


    /*
     * Do all the things here!
     **/

    std::cout << "Took " << getTime() << "ms.\n";
}
