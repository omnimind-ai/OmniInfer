#ifndef CODEGEN_H
#define CODEGEN_H

#include "symbols.h"
#include "parser.h"

typedef struct fb_output fb_output_t;

struct fb_output {
    /*
     * Common namespace across files. May differ from namespace
     * for consistent use of type names.
     */
    char nsc[FLATCC_NAMESPACE_MAX + 2];
    char nscup[FLATCC_NAMESPACE_MAX + 2];

    FILE *fp;
    fb_schema_t *S;
    fb_options_t *opts;
    fb_scope_t *current_scope;
    int indent;
    int spacing;
    int tmp_indent;
};

int __flatcc_fb_init_output_c(fb_output_t *out, fb_options_t *opts);
#define fb_init_output_c __flatcc_fb_init_output_c
void __flatcc_fb_end_output_c(fb_output_t *out);
#define fb_end_output_c __flatcc_fb_end_output_c

int __flatcc_fb_codegen_common_c(fb_output_t *out);
#define fb_codegen_common_c __flatcc_fb_codegen_common_c

int __flatcc_fb_codegen_c(fb_output_t *out, fb_schema_t *S);
#define  fb_codegen_c __flatcc_fb_codegen_c

void *__flatcc_fb_codegen_bfbs_to_buffer(fb_options_t *opts, fb_schema_t *S, void *buffer, size_t *size);
#define fb_codegen_bfbs_to_buffer __flatcc_fb_codegen_bfbs_to_buffer

void *__flatcc_fb_codegen_bfbs_alloc_buffer(fb_options_t *opts, fb_schema_t *S, size_t *size);
#define fb_codegen_bfbs_alloc_buffer __flatcc_fb_codegen_bfbs_alloc_buffer

int __flatcc_fb_codegen_bfbs_to_file(fb_options_t *opts, fb_schema_t *S);
#define fb_codegen_bfbs_to_file __flatcc_fb_codegen_bfbs_to_file

#endif /* CODEGEN_H */
