#include <assert.h>
#include <sys/time.h>
#include <stdio.h>

//#define INT_SET_PRIVATE
#ifdef INT_SET_PRIVATE
/* Make all hash functions private to this module for better
 * performance. This may not be necessary depending on compiler
 * optimizations. clang 4.2 -O3 benefits while -O4 figures it and get
 * same speed with external linkage. */
#define HT_PRIVATE
#include "int_set.h"
#include "ptr_set.c"
#undef HT_PRIVATE
#else
/* Use external linkage. Link with ptr_set.c which int_set depends upon. */
#include "int_set.h"
#endif

struct timeval time_diff(struct timeval start, struct timeval end)
{
    struct timeval temp;
    if ((end.tv_usec-start.tv_usec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_usec = 1000000+end.tv_usec-start.tv_usec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_usec = end.tv_usec-start.tv_usec;
    }
    return temp;
}

double elapsed_ms(struct timeval td)
{
    return (double)td.tv_sec * 1000 + (double)td.tv_usec / 1000;
}

void test_int_set()
{
    int i, x;
    const int N = 1000000;
    //const int N = 1000;
    int_set_t ht = {0};
    int_set_t *S = &ht;
    double ms, nsop, opms;
    struct timeval t1, t2, td;

    for (i = 1; i <= N; ++i) {
        int_set_add(S, i);
        assert(int_set_exists(S, i));
    }
    assert(int_set_count(S) == N);

    for (i = 1; i <= N; ++i) {
        assert(int_set_exists(S, i));
    }

    gettimeofday(&t1, 0);
    for (x = 0, i = 1; i <= N; ++i) {
        x += int_set_exists(S, i);
    }
    gettimeofday(&t2, 0);

    td = time_diff(t1, t2);
    ms = elapsed_ms(td);

    nsop = ms * 1000000 / x;
    opms = (double)x / ms;
    printf("%d out of %d keys found in time %0.03f ms or %0.01f ns per op\n",
            x, N, ms, nsop);
    printf("ops / ms: %0.0f\n", opms);

    for (i = 1; i <= N; ++i) {
        assert(int_set_count(S) == N - i + 1);
        assert(int_set_exists(S, i));
        int_set_remove(S, i);
        assert(!int_set_exists(S, i));
    }
    assert(int_set_count(S) == 0);
}

int main(int argc, char *argv[])
{
    test_int_set();
    return 0;
}
