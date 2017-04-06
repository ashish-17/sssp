#ifndef	USER_SPECIFIED_STRUCTURES_H
#define	USER_SPECIFIED_STRUCTURES_H

/**************************************
 *  STRUCTURES
 **************************************/


// Vertex structure.
struct Vertex{

	unsigned int distance;
};

// Vertex_static structure. Those properties of the vertex that remain constant during processing should be declared here.
typedef struct Vertex_static{

}Vertex_static;

// Edge structure.
struct Edge{

	unsigned int weight;

};

typedef struct GraphEdge {
  unsigned int weight;
  unsigned int src;
  unsigned int dest;
} GraphEdge_t;

#endif	//	USER_SPECIFIED_STRUCTURES_H
