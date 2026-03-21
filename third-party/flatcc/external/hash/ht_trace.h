#ifndef HT_TRACE_H
#define HT_TRACE_H

#ifdef HT_TRACE_ON
#ifndef HT_TRACE_OUT
#define HT_TRACE_OUT stderr
#endif

#include <stdio.h>
#define ht_trace(s) fprintf(HT_TRACE_OUT, "trace: %s\n", s)
#define ht_tracei(s, i) fprintf(HT_TRACE_OUT, "trace: %s: %d\n", s, (int)i)
#define ht_tracex(s, x) fprintf(HT_TRACE_OUT, "trace: %s: 0x%lx\n", s, (long)x)
#define ht_traces(s, s2, len) fprintf(HT_TRACE_OUT, "trace: %s: %.*s\n", s, (int)len, s2)

static void ht_trace_buckets(hash_table_t *ht, char *msg, int first, int count)
{
    int i, j, N, n;

    n = ht->buckets;
    N = n - 1;

    if (count == 0) {
        count = 32;
    }
    if (count > n) {
        count = n;
    }

    first = first & N;
    fprintf(HT_TRACE_OUT, "bucket trace: %s\n", msg);
    if (n > count) {
        n = count;
    }
    fprintf(HT_TRACE_OUT, "item count: %ld, bucket count %ld, utilization: %0.1f%%\n",
            ht->count, ht->buckets, (double)ht->count / ht->buckets * 100);

    if (ht->offsets) {
        for (i = 0; i < n; ++i) {
            j = (first + i) & N;
            fprintf(HT_TRACE_OUT, "%03d:%08x:[%02d]\n",
                    j, (unsigned int)((void **)ht->table)[j], (unsigned int)ht->offsets[j]);
        }
    } else {
        for (i = 0; i < n; ++i) {
            j = (first + i) & N;
            fprintf(HT_TRACE_OUT, "%03d:%08x\n", j, (unsigned int)((void **)ht->table)[j]);
        }
    }
    fprintf(HT_TRACE_OUT, "--\n");
}
#else
#define ht_trace(arg1) ((void)0)
#define ht_tracei(arg1, arg2) ((void)0)
#define ht_tracex(arg1, arg2) ((void)0)
#define ht_traces(arg1, arg2, arg3) ((void)0)
#define ht_trace_buckets(arg1, arg2, arg3, arg4) ((void)0)
#endif

#endif /* HT_TRACE_H */
