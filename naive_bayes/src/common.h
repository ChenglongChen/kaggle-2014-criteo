#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _UTIL_H_
#define _UTIL_H_

#define flag { printf("\nLINE: %d\n", __LINE__); fflush(stdout); }

#include <cstdio>
#include <string>

typedef unsigned int uint;
typedef unsigned long long ull;

FILE *open_c_file(std::string const &path, std::string const &mode);

#endif // _UTIL_H_
