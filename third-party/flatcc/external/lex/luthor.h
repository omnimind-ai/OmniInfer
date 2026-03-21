/*
 * Mostly generic lexer that can be hacked to suit specific syntax. See
 * more detailed comments further down in this file.
 *
 * Normally include luthor.c instead of luthor.h so emitter functions
 * can be custom defined, and optionally also fast keyword definitions.
 *
 * At the very minimum, define lex_emit which other emitters default to.
 *
 * Create a wrapper function to drive the lex function in said file.
 *
 * Use this header in separate parser logic to access the token values
 * if relevant.
 */

#ifndef LUTHOR_H
#define LUTHOR_H

#ifdef LEX_KEYWORDS
#include <string.h> /* memcmp for kw match */
#endif

#include "tokens.h"

#ifndef lex_emit
#define lex_emit(token, first, last) ((void)0)
#endif

/*
 * Default for comments, bom, and other things that are not necessarily
 * of interest to the parser, but may be to buffer wrap handling,
 * debugging, and pretty printers.
 */
#ifndef lex_emit_other
#define lex_emit_other(token, first, last) ((void)0)
#endif

#ifndef lex_emit_eof
#define lex_emit_eof(pos) lex_emit(LEX_TOK_EOF, pos, pos)
#endif

#ifndef lex_emit_abort
#define lex_emit_abort(pos) lex_emit(LEX_TOK_ABORT, pos, pos)
#endif

#ifndef lex_emit_eob
#define lex_emit_eob(pos) lex_emit(LEX_TOK_EOB, pos, pos)
#endif

#ifndef lex_emit_eos
#define lex_emit_eos(first, last) lex_emit(LEX_TOK_EOS, first, last)
#endif

#ifndef lex_emit_bom
#define lex_emit_bom(first, last) lex_emit_other(LEX_TOK_BOM, first, last)
#endif

#ifndef lex_emit_id
#ifdef LEX_KEYWORDS
/* LEX_KW_TABLE_BEGIN .. LEX_KEYWORD_TABLE_END defines lex_match_kw. */
#define lex_emit_id(first, last, tag) lex_emit(lex_match_kw(tag, first), first, last)
#else
#define lex_emit_id(first, last, tag) lex_emit(LEX_TOK_ID, first, last)
#endif
#endif

/*
 * This is a default for unknown symbols. It may be treated as an error,
 * or it can be processed further by the parser instead of customizing
 * the lexer. It ensures that there is always a token for every part of
 * the input stream.
 */
#ifndef lex_emit_symbol
#define lex_emit_symbol(token, first, last) lex_emit(LEX_TOK_SYMBOL, first, last)
#endif

/*
 * Control characters 0x01 .. 0x1f, 0x7f(DEL), excluding \0\r\n\t which have
 * separate tokens.
 *
 * Control characters in strings and comments are passed on as body
 * elements, except \0\r\n which breaks the string up.
 */
#ifndef lex_emit_ctrl
#define lex_emit_ctrl(pos) lex_emit(LEX_TOK_CTRL, pos, pos + 1)
#endif

#ifndef lex_emit_string_ctrl
#define lex_emit_string_ctrl(pos) lex_emit(LEX_TOK_STRING_CTRL, pos, pos + 1)
#endif

#ifndef lex_emit_comment_ctrl
#define lex_emit_comment_ctrl(pos) lex_emit_other(LEX_TOK_COMMENT_CTRL, pos, pos + 1)
#endif

/*
 * This enables user to both count lines, and to calculate character
 * offset for subsequent lexemes. New line starts a lexeme, line break
 * symbol is located at lexeme - skipped and with have length 2 if \r\n
 * or \n\r break, and 1 otherwise.
 */
#ifndef lex_emit_newline
#define lex_emit_newline(first, last) lex_emit(LEX_TOK_NEWLINE, first, last)
#endif

#ifndef lex_emit_string_newline
#define lex_emit_string_newline(first, last) lex_emit(LEX_TOK_STRING_NEWLINE, first, last)
#endif

#ifndef lex_emit_int
#define lex_emit_int(first, last) lex_emit(LEX_TOK_INT, first, last)
#endif

#ifndef lex_emit_float
#define lex_emit_float(first, last) lex_emit(LEX_TOK_FLOAT, first, last)
#endif

#ifndef lex_emit_int_suffix
#define lex_emit_int_suffix(first, last) lex_emit(LEX_TOK_INT_SUFFIX, first, last)
#endif

#ifndef lex_emit_float_suffix
#define lex_emit_floatint_suffix(first, last) lex_emit(LEX_TOK_FLOAT_SUFFIX, first, last)
#endif

#ifndef lex_emit_binary
#define lex_emit_binary(first, last) lex_emit(LEX_TOK_BINARY, first, last)
#endif

#ifndef lex_emit_octal
#define lex_emit_octal(first, last) lex_emit(LEX_TOK_OCTAL, first, last)
#endif

#ifndef lex_emit_hex
#define lex_emit_hex(first, last) lex_emit(LEX_TOK_HEX, first, last)
#endif

#ifndef lex_emit_hex_float
#define lex_emit_hex_float(first, last) lex_emit(LEX_TOK_HEX_FLOAT, first, last)
#endif

/*
 * The comment token can be used to aid backtracking during buffer
 * switch.
 */
#ifndef lex_emit_comment_begin
#define lex_emit_comment_begin(first, last, is_doc)                         \
    lex_emit_other(LEX_TOK_COMMENT_BEGIN, first, last)
#endif

#ifndef lex_emit_comment_part
#define lex_emit_comment_part(first, last) lex_emit_other(LEX_TOK_COMMENT_PART, first, last)
#endif

#ifndef lex_emit_comment_end
#define lex_emit_comment_end(first, last) lex_emit_other(LEX_TOK_COMMENT_END, first, last)
#endif

#ifndef lex_emit_comment_unterminated
#define lex_emit_comment_unterminated(pos)                                  \
        lex_emit_other(LEX_TOK_COMMENT_UNTERMINATED, pos, pos)
#endif

#ifndef lex_emit_comment_deeply_nested
#define lex_emit_comment_deeply_nested(pos)                                 \
        lex_emit_other(LEX_TOK_COMMENT_DEEPLY_NESTED, pos, pos)
#endif

#ifndef lex_emit_string_begin
#define lex_emit_string_begin(first, last) lex_emit(LEX_TOK_STRING_BEGIN, first, last)
#endif

#ifndef lex_emit_string_part
#define lex_emit_string_part(first, last) lex_emit(LEX_TOK_STRING_PART, first, last)
#endif

#ifndef lex_emit_string_end
#define lex_emit_string_end(first, last) lex_emit(LEX_TOK_STRING_END, first, last)
#endif

#ifndef lex_emit_string_escape
#define lex_emit_string_escape(first, last) lex_emit(LEX_TOK_STRING_ESCAPE, first, last)
#endif

#ifndef lex_emit_string_unterminated
#define lex_emit_string_unterminated(pos)                                   \
        lex_emit(LEX_TOK_STRING_UNTERMINATED, pos, pos)
#endif

#ifndef lex_emit_blank
#define lex_emit_blank(first, last)                                         \
        lex_emit_other(LEX_TOK_BLANK, first, last)
#endif

#ifndef lex_emit_op
#define lex_emit_op(op, first, last)  lex_emit((long)(op), first, last)
#endif

#ifndef lex_emit_compound_op
#define lex_emit_compound_op(op1, op2, first, last)                         \
        lex_emit(((long)(op1) | ((long)(op2) << 8)), first, last)
#endif

#ifndef lex_emit_tricompound_op
#define lex_emit_tricompound_op(op1, op2, op3, first, last)                 \
        lex_emit(((long)(op1) | ((long)(op2) << 8)) |                       \
                ((long)(op3)<<16), first, last)
#endif

#ifndef lex_emit_quadcompound_op
#define lex_emit_quadcompound_op(op1, op2, op3, op4, first, last)           \
        lex_emit(((long)(op1) | ((long)(op2) << 8)) |                       \
                ((long)(op3) << 16) | ((long)(op4) << 24), first, last)
#endif

/* Used to limit number of nested comment level. */
#ifndef LEX_MAX_NESTING_LEVELS
#define LEX_MAX_NESTING_LEVELS 100
#endif


/* Keyword handling macros, see `keywords.c` for an example usage. */
#ifdef LEX_KEYWORDS

/*
 * This implements a switch statement branching on the 4 character
 * keyword tag (unsigned long value) which is produced by the lexers id
 * recognizer. A final check is needed with to ensure an exact
 * match with a given id. Two keywords rarely conflicts, but it is
 * possible, and therefore kw_begin kw_match kw_match ... kw_end is used
 * to cover this.
 *
 * See example usage elsewhere for details.
 *
 * The first element x0 is length '0'..'9' and ensure comparisons will
 * not overrun the buffer where the lexeme is stored during string
 * comparison, iff the keywords report the length correctly.
 *
 * The next elements in the tag are the first, second, and last
 * character of lexeme / keyword, replacing second character with '\0'
 * on single length keywords, so keyword 'e' is tagged '1', 'e', '\0', 'e',
 * and 'while' is tagged '5' 'w', 'h', 'e', where the length is lsb
 * and last chararacter is msb.
 *
 * An enum with tok_kw_<name> elements is expected to provide return
 * values on match. These should start at LEX_TOK_KW_BASE and are
 * negative.
 *
 */
#define lex_kw_begin(x0, x1, x2, x3)                                    \
        case                                                            \
            ((unsigned long)(x0) |                                      \
            ((unsigned long)(x1) << 8) |                                \
            ((unsigned long)(x2) << 16) |                               \
            ((unsigned long)(x3) << 24)) :

#define lex_kw_match(kw)                                                \
        if (memcmp(#kw, lexeme, sizeof(#kw) - 1) == 0)                  \
                return tok_kw_##kw;

#define lex_kw_end()                                                    \
        break;

#define lex_kw(kw, x0, x1, x2, x3)                                      \
        lex_kw_begin(x0, x1, x2, x3)                                    \
            lex_kw_match(kw)                                            \
        lex_kw_end()

static long lex_match_kw(unsigned long tag, const char *lexeme);

/* Static so multiple grammers are possible in a single program. */
#define LEX_KW_TABLE_BEGIN                                              \
static long lex_match_kw(unsigned long tag, const char *lexeme)         \
{                                                                       \
    switch (tag) {                                                      \

#define LEX_KW_TABLE_END                                                \
    default:                                                            \
        break;                                                          \
    }                                                                   \
    return LEX_TOK_KW_NOT_FOUND;                                        \
}

#else

/* Allow flagging in and out without unused warning or missing macros */
#define lex_kw_begin(x0, x1, x2, x3)
#define lex_kw_match(kw)
#define lex_kw_end()
#define lex_kw(kw, x0, x1, x2, x3)
#define LEX_KEYWORD_TABLE_BEGIN
#define LEX_KEYWORD_TABLE_END

#endif /* LEX_KEYWORDS */



/*
 * Modes used for recovery when switching to a new buffer and handling
 * internal state changes for strings and comments.
 */
enum {
    /* Always 0, is initial lexer state. */
    LEX_MODE_NORMAL = 0,

    /* Returned if lex is given unsupported mode. */
    LEX_MODE_INVALID = 1,

    /*
     * Can be used in place of normal mode to consume optional bom
     * marker at buffer start. Only utf-8 bom is supported.
     */
    LEX_MODE_BOM,

    /*
     * Returned at end of buffer if mid string or mid comment, may also
     * be larger for nested comments as nesting level is encoded.
     */
    LEX_MODE_C_STRING,
    LEX_MODE_C_STRING_SQ,
    LEX_MODE_PYTHON_BLOCK_STRING,
    LEX_MODE_PYTHON_BLOCK_STRING_SQ,
    LEX_MODE_C_BLOCK_COMMENT,
    LEX_MODE_LINE_COMMENT,
    LEX_MODE_JULIA_NESTED_COMMENT,


    /* Counter embedded in mode. */
    LEX_MODE_COUNT_BASE = 16,
};



/* ON CALLING AND USING LEX FUNCTION
 *
 * If utf-8 BOM possible, detect this before calling the lexer and
 * advance the buffer. JSON explititly disallows BOM, but recommends
 * consuming it if present. If some other Unicode BOM is found, convert
 * the buffer first. The lexer assumes ALL non-ascii characters are
 * valid trailing identifiers which mostly works well. Strings with
 * broken utf-8 are passed on as is. utf-8 identifiers must be enabled
 * with #define LEX_ENABLE_UTF8_ID
 *
 * If required, postprocess identifiers and strings for valid utf-8.  It
 * is assumed that all keywords are at most 9 characters long and always
 * ASCII. Otherwise post process them in a hash table on identifier
 * event. This enables a fast compiled trie lookup of keywords.
 *
 * Newline and control characters are always emitted, also inside
 * strings and comments. The exception is \r, \n, \t, \0 which are
 * handled specially, or if the lexer is adapted to handle certain
 * control characters specially.
 *
 * Each token is not guaranteed correct, only to be delimited correct,
 * if it is indeed correct. Only very few tokens can be zero length, for
 * example, the parser can rely on string part token not being empty
 * which is important in dealing with line continuation. The end of
 * buffer token is empty, and so is the unterminates string token, and
 * also the comment end token for single line tokens, but not the
 * multi-line version. There is a token for every part of the input
 * stream, but the parser can easily define some to be ignored and have
 * them optimized out.
 *
 * Strings have start token, and optionally sequences of control,
 * escape, and newline tokens, followed by either string end token or
 * string unterminated token. Strings delimiters can be one
 * (single-line) or three double quotes (multi-line, like python, but
 * cannot be single quotes, unlike Python. Python, C and Javascript
 * string continuation is handled by having the parser observing string
 * escape followed by newline token. Escape is always a single
 * character '\' token, and the parser is responsible for consuming the
 * following content. If string syntax with double delimiter is used to
 * define escaped delimiter, this will occur as two separate strings
 * with no space between. The parser can handle this on its own; if, in
 * such strings, '\"' does not mean escaped delimiter, the string will
 * not terminate correctly, and the lexer must be apapted. Unterminated
 * string may happen at end of buffer, also for single line comments.
 * This is because the string might continue in a new buffer. The parser
 * should deal with this.
 *
 * Comments always start with a start token, followed by zero or more
 * comment part tokens interleaved with control and newline tokens,
 * terminated by either comment end token, or unterminated comment
 * token. If the comment is single, the unterminated comment token may
 * appear at the last line instead of the expected end of comment token
 * because the comment might continue in a new buffer. The parser
 * should deal with this. Escapes and line continuations have no effects
 * in comments, unlike strings.
 *
 * The lexer only carries one state variable: the mode. The mode can be
 * normal (default and equals zero), or single or multi string or
 * comment modes.  These modes are used to to recover after switching
 * buffers as discussed below.
 *
 * The lexer can run to completion without involving the parser and
 * could be used to pipeline tokens into another thread for concurrent
 * parsing which is safe since the input buffer is considered read-only.
 *
 *
 * KEYWORDS
 *
 * Keywords are treated as identifiers by default. By including a
 * keyword table the `lex_emit_id` macro will check if the id is a
 * keyword and translate the token if it is. Using the provided keyword
 * table macros is just one way to do it. This is better explained by
 * looking at an example. Keyword lookup based on the precomputed keyword
 * tag provided to the lookup function are limited to 9 characters, but a
 * custom lookup function need not use it and then the tag precomputation
 * will be optimized out.
 *
 * Keywords are defined by the lookup function and should be negative
 * starting at LEX_TOK_KW_BASE to avoid conflicts with other token types.
 *
 *
 * WRAPPING MULTIPLE BUFFERS
 *
 * The user may need to deal with multiple buffers because data may
 * arrive asynchronously over the network, and may have many concurrent
 * lexing jobs. The emitter part is not difficult since a ring buffer
 * can grow, or the parser can be called directly (except queuing a few
 * tokens for backtracking as we shall see).
 *
 * If the lexer were an explicit statemachine as in Flex, we could get
 * an yywrap event to fill buffers, but our state is on the stack and in
 * registers for optimization. We may use co-routines, but it doesn't
 * cover all issues, and, as it turns out is not necessary with the
 * following restrictions on syntax:
 *
 * All variable length tokens such as numerics and identifiers are
 * limited in length. Strings and comments are not, but are broken into
 * zero, one, or several body tokens per line. ANSI-C limits line length
 * to 509 characters (allowing for continuation and two byte linebreaks
 * in a 512 byte buffer). But JSON has no line continuation for strings
 * and may (and often do) store everything on a single line. Whitespace
 * can also extend beyond given limit.
 *
 * If we ignore whitespace, strings and comments, we can discard the
 * last token (or last two in case there are paired tokens, such as
 * leading zero followed by numeric. Parsing can then resume in a new
 * buffer where the first 512 bytes (or similar) are duplicated from the
 * previous buffer. The lexer is then restarted at the last token (pair)
 * start which may turn out to change the length or even introduce a
 * different result such introducing leading zero. The lexer need no
 * specific state to do this.
 *
 * For strings and comments, we need a flag to allow entering the lexer
 * mid string or mid comment. The newline and line continuation tokens
 * need to be dropped, and the last body may need to be truncated as it
 * can embed a partial delimiter. The simplest way to deal with this is
 * to backtrack tokens until the last token begins at a safe position,
 * about 3-6 charaters earlier, and truncating body segments that span
 * this barrier. Whitespace can also be truncated.
 *
 * We can generalize this further by going at least K bytes back in an N
 * overlap buffer region and require non-strings (and non-comments) to
 * not exceed N-K bytes, where K and N are specific to the syntax and
 * the I/O topology.
 *
 * We can add flags to tokens that can help decide how to enter
 * backtracking mode without covering every possible scanner loop - i.e.
 * are we mid string, mid comment, single-line or multi-line.
 *
 * All the lexer needs to do then, is to receive the backtracking mode
 * flags. A wrapping driver can deal with backtrack logic, which is
 * specific to how tokens are emitted. Whitespace need no recovery mode
 * but perhaps new whitespace should extend existing to simplify
 * parsing.
 */


#endif /* LUTHOR_H */

