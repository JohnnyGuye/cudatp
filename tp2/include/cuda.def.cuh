#pragma once

#include <iostream>
#define CHECK_ERROR { auto err = cudaGetLastError(); std::cerr << cudaGetErrorName( err ) << " " << __LINE__ << " " << __FUNCTION__ << std::endl; }
#define GCPU_F __host__ __device__
