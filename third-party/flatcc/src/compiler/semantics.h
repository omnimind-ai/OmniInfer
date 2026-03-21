#ifndef SCHEMA_H
#define SCHEMA_H

#include "parser.h"

int __flatcc_fb_build_schema(fb_parser_t *P);
#define fb_build_schema __flatcc_fb_build_schema


fb_scope_t *fb_find_scope(fb_schema_t *S, fb_ref_t *name);

#endif /* SCHEMA_H */
