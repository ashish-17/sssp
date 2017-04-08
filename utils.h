#ifndef UTILS_H
#define UTILS_H

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#define SSSP_INF 1073741824
#define D_INFINITY SSSP_INF

#ifdef __cplusplus
extern "C" {
#endif
void setTime();
double getTime();
void mergeSortSeq(void* data, int item_size, int n, int (*comparator)(void*, void*));
#ifdef __cplusplus
}
#endif

// Open files safely.
template <typename T_file>
void openFileToAccess( T_file& input_file, std::string file_name ) {
	input_file.open( file_name.c_str() );
	if( !input_file )
		throw std::runtime_error( "Failed to open specified file: " + file_name + "\n" );
}
#endif	//	SIMPLETIMER_H
