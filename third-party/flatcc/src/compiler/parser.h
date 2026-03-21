#ifndef PARSER_H
#define PARSER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "../../config/config.h"
#include "flatcc/flatcc.h"
#include "symbols.h"

#define ELEM_BUFSIZ (64 * 1024)
#define ERROR_BUFSIZ 200

#define REVERT_LIST(TYPE, FIELD, HEAD)                                      \
    do {                                                                    \
        TYPE *tmp__next, *tmp__prev = 0, *tmp__link = *(HEAD);              \
        while (tmp__link) {                                                 \
            tmp__next = tmp__link->FIELD;                                   \
            tmp__link->FIELD = tmp__prev;                                   \
            tmp__prev = tmp__link;                                          \
            tmp__link = tmp__next;                                          \
        }                                                                   \
        *(HEAD) = tmp__prev;                                                \
    } while (0)

typedef struct fb_parser fb_parser_t;
typedef flatcc_options_t fb_options_t;

typedef void (*fb_error_fun)(void *err_ctx, const char *buf, size_t len);

void __flatcc_fb_default_error_out(void *err_ctx, const char *buf, size_t len);
#define  fb_default_error_out __flatcc_fb_default_error_out

int __flatcc_fb_print_error(fb_parser_t *P, const char * format, ...);
#define fb_print_error __flatcc_fb_print_error

struct fb_parser {
    fb_parser_t *dependencies;
    fb_parser_t *inverse_dependencies;
    fb_error_fun error_out;
    void *error_ctx;

    const char *managed_input;

    fb_token_t *ts, *te;
    int tcapacity;
    int doc_mode;
    fb_doc_t *doc;
    fb_token_t *token;

    size_t elem_end;
    void *elem_buffers;
    size_t elem;
    size_t offset_size;

    const char *line;
    long linenum;

    /* Internal id (not a pointer into token stream). */
    fb_token_t t_none;
    fb_token_t t_ubyte;

    int failed;

    unsigned char *tmp_field_marker;
    fb_symbol_t **tmp_field_index;
    int nesting_level;

    int has_schema;
    fb_options_t opts;
    fb_schema_t schema;
    fb_scope_t *root_scope;
    fb_scope_t *current_scope;
    char *path;
    char *referer_path;
};

static inline void checkmem(const void *p)
{
    if (!p) {
        fprintf(stderr, "error: out of memory, aborting...\n");
        exit(1);
    }
}

static inline void *new_elem(fb_parser_t *P, size_t size)
{
    size_t elem;
    void *buf;

    size = (size + 15) & ~(size_t)15;
    elem = P->elem;
    if (elem + size > P->elem_end) {
        buf = calloc(ELEM_BUFSIZ, 1);
        checkmem(buf);
        *(void**)buf = P->elem_buffers;
        P->elem_buffers = buf;
        elem = P->elem = (size_t)buf + 16;
        P->elem_end = (size_t)buf + ELEM_BUFSIZ;
    }
    P->elem += size;
    return (void*)elem;
}

int __flatcc_fb_print_error(fb_parser_t *P, const char * format, ...);
#define fb_print_error __flatcc_fb_print_error

const char *__flatcc_error_find_file_of_token(fb_parser_t *P, fb_token_t *t);
#define error_find_file_of_token __flatcc_error_find_file_of_token

/*
 * This is the primary error reporting function.
 * The parser is flagged as failed and error count incremented.
 *
 * If s is not null, then s, len replaces the token text of `t` but
 * still reports the location of t. `peer` is optional and prints the
 * token location and text and the end of the message.
 * `msg` may be the only non-zero argument besides `P`.
 *
 * Various helper functions are available for the various cases.
 *
 * `fb_print_error` may be called instead to generate text to the error
 * output that is not counted as an error.
 */
void __flatcc_error_report(fb_parser_t *P, fb_token_t *t, const char *msg, fb_token_t *peer, const char *s, size_t len);
#define error_report __flatcc_error_report

void __flatcc_warn_report(fb_parser_t *P, fb_token_t *t, const char *msg, fb_token_t *peer, const char *s, size_t len);
#define warn_report __flatcc_warn_report

static void error_tok_2(fb_parser_t *P, fb_token_t *t, const char *msg, fb_token_t *peer)
{
    error_report(P, t, msg, peer, 0, 0);
}

static inline void error_tok(fb_parser_t *P, fb_token_t *t, const char *msg)
{
    error_tok_2(P, t, msg, 0);
}

static inline void warn_tok(fb_parser_t *P, fb_token_t *t, const char *msg)
{
    warn_report(P, t, msg, 0, 0, 0);
}

/* Show both token and string */
static inline void warn_tok_as_string(fb_parser_t *P, fb_token_t *t, const char *msg, const char *s, size_t len)
{
    warn_report(P, t, msg, 0, s, len);
}

/* Only use the token location. */
static inline void error_tok_as_string(fb_parser_t *P, fb_token_t *t, const char *msg, char *s, size_t len)
{
    error_report(P, t, msg, 0, s, len);
}

static inline void error(fb_parser_t *P, const char *msg)
{
    error_tok(P, 0, msg);
}

static inline void error_name(fb_parser_t *P, fb_name_t *name, const char *msg)
{
    if (!name) {
        error(P, msg);
    } else {
        error_report(P, 0, msg, 0, name->name.s.s, (size_t)name->name.s.len);
    }
}

static inline void error_sym(fb_parser_t *P, fb_symbol_t *s, const char *msg)
{
    error_tok(P, s->ident, msg);
}

static inline void error_sym_2(fb_parser_t *P, fb_symbol_t *s, const char *msg, fb_symbol_t *s2)
{
    error_tok_2(P, s->ident, msg, s2->ident);
}

static inline void error_sym_tok(fb_parser_t *P, fb_symbol_t *s, const char *msg, fb_token_t *t2)
{
    error_tok_2(P, s->ident, msg, t2);
}

void error_ref_sym(fb_parser_t *P, fb_ref_t *ref, const char *msg, fb_symbol_t *s2);

static inline void error_ref(fb_parser_t *P, fb_ref_t *ref, const char *msg)
{
    error_ref_sym(P, ref, msg, 0);
}

/*
 * If `opts` is null, defaults options are being used, otherwise opts is
 * copied into the parsers options. The name may be path, the basename
 * without default extension will be extracted. The `error_out` funciton is
 * optional, otherwise output is printed to stderr, truncated to a
 * reasoanble size per error. `error_ctx` is provided as argument to
 * `error_out` if non-zero, and otherwise ignored.
 *
 * This api only deals with a single schema file so a parent level
 * driver must handle file inclusion and update P->dependencies but
 * order is not significant (parse order is, but this is handled by
 * updating the `include_index` in the root schema).
 *
 * P->dependencies must be cleared by callee in any order but once one
 * is cleared the entire structure should be taken down because symbols
 * trees point everywhere. For parses without file inclusion
 * dependencies will be null. Dependencies are not handled at this
 * level. P->inverse_dependencies is just the reverse list.
 *
 * The file at the head of the dependencies list is the root and the
 * one that provides the root schema. Other root schemas are not used.
 */
int __flatcc_fb_init_parser(fb_parser_t *P, fb_options_t *opts, const char *name,
        fb_error_fun error_out, void *error_ctx, fb_root_schema_t *rs);
#define fb_init_parser __flatcc_fb_init_parser

int __flatcc_fb_parse(fb_parser_t *P, const char *input, size_t len, int own_buffer);
#define fb_parse __flatcc_fb_parse

void __flatcc_fb_clear_parser(fb_parser_t *P);
#define fb_clear_parser __flatcc_fb_clear_parser

#endif /* PARSER_H */
