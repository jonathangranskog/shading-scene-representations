#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#ifdef __CUDACC__
    #define IS_CUDA 1
#endif

#ifndef RT_FUNCTION
    #ifdef __CUDACC__
        #define RT_FUNCTION __forceinline__ __device__
    #else
        #define RT_FUNCTION inline
    #endif
#endif

#endif