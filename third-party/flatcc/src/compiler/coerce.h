#ifndef COERCE_H
#define COERCE_H

#include <assert.h>

#include "symbols.h"
#include "parser.h"

int __flatcc_fb_coerce_scalar_type(fb_parser_t *P,
        fb_symbol_t *sym, fb_scalar_type_t st, fb_value_t *value);
#define fb_coerce_scalar_type __flatcc_fb_coerce_scalar_type

#endif /* COERCE_H */
