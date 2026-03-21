/*
 * Designed to be included in other C files which define emitter
 * operations. The same source may thus be used to parse different
 * grammars.
 *
 * The operators cover the most common operators i the C family.  Each
 * operator does not have a name, it is represent by a long token code
 * with up to 4 ASCII characters embedded literally. This avoids any
 * semantic meaning at the lexer level. Emitters macros can redefine
 * this behavior.
 *
 * No real harm is done in accepting a superset, but the source is
 * intended to be modified, have things flagged or removed, other things
 * added. The real complicity is in numbers, identifiers, and comments,
 * which should be fairly complete with flagging as is.
 *
 * Keyword handling is done at macroes, and described elsewhere, but for
 * identifier compatible keywords, this is quite efficient to handle on
 * a per language basis without modifying this source.
 *
 * The Lisp language family is somewhat different and not directly
 * suited for this lexer, although it can easily be modified to suit.
 * The main reason is ';' for comments, and operators used as part of
 * the identifier symbol set, and no need for operator classification,
 * and different handling of single character symbols.
 *
 * So overall, we more or less have one efficient unified lexer that can
 * manage many languages - this is good, because it is a pain to write a
 * new lexer by hand, and lexer tools are what they are.
 */

#include "luthor.h"

#ifdef LEX_C99_NUMERIC
#define LEX_C_NUMERIC
#define LEX_HEX_FLOAT_NUMERIC
#define LEX_BINARY_NUMERIC
#endif

#ifdef LEX_C_NUMERIC
#define LEX_C_OCTAL_NUMERIC
#define LEX_HEX_NUMERIC
#endif

#ifdef LEX_JULIA_NUMERIC
#ifdef LEX_C_OCTAL_NUMERIC
/*
 * LEX_JULIA_OCTAL_NUMERIC and LEX_C_OCTAL_NUMERIC can technically
 * coexist, but leading zeroes give C style leading zero numbers
 * which can lead to incorrect values depending on expectations.
 * Therefore the full LEX_JULIA_NUMERIC flag is designed to not allow this.
 */
#error "LEX_C_OCTAL_NUMERIC conflicts with LEX_JULIA_NUMERIC leading zero integers"
#endif

/*
 * Julia v0.3 insists on lower case, and has a different meaning for
 * upper case.
 */
#define LEX_LOWER_CASE_NUMERIC_PREFIX
#define LEX_JULIA_OCTAL_NUMERIC
#define LEX_HEX_FLOAT_NUMERIC
#define LEX_BINARY_NUMERIC

#endif

#ifdef LEX_HEX_FLOAT_NUMERIC
#define LEX_HEX_NUMERIC
#endif

/*
 * Numeric and string constants do not accept prefixes such as u, l, L,
 * U, ll, LL, f, or F in C, or various others in Julia strings. Use the
 * parser to detect juxtaposition between identifier and constant. In
 * Julia numeric suffix means multiplication, in C it is a type
 * qualifier.  Sign, such as defined in JSON, are also not accepted -
 * they must be operators.  See source for various flag to enable
 * different token types.
 */

/*
 * Includes '_' in identifers by default. Defines follow characters in
 * identifiers but not the lead character - it must be defined in switch
 * cases.  If the identifier allows for dash '-', it is probably better
 * to handle it as an operator and flag surrounding space in the parser.
 */
#ifndef lex_isalnum

/*
 * NOTE: isalnum, isalpha, is locale dependent. We only want to
 * to consider that ASCII-7 subset and treat everything else as utf-8.
 * This table is not for leading identifiers, as it contains 0..9.
 *
 * For more correct handling of UTF-8, see:
 * https://theantlrguy.atlassian.net/wiki/display/ANTLR4/Grammar+Lexicon
 * based on Java Ident = NameStartChar NameChar*
 *
 * While the following is UTF-16, it can be adapted to UTF-8 easily.


    fragment
    NameChar
       : NameStartChar
       | '0'..'9'
       | '_'
       | '\u00B7'
       | '\u0300'..'\u036F'
       | '\u203F'..'\u2040'
       ;
    fragment
    NameStartChar
       : 'A'..'Z' | 'a'..'z'
       | '\u00C0'..'\u00D6'
       | '\u00D8'..'\u00F6'
       | '\u00F8'..'\u02FF'
       | '\u0370'..'\u037D'
       | '\u037F'..'\u1FFF'
       | '\u200C'..'\u200D'
       | '\u2070'..'\u218F'
       | '\u2C00'..'\u2FEF'
       | '\u3001'..'\uD7FF'
       | '\uF900'..'\uFDCF'
       | '\uFDF0'..'\uFFFD'
       ;
 */

static const char lex_alnum[256] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    /* 0..9 */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    /* A..O */
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    /* P..Z, _ */
#ifdef LEX_ID_WITHOUT_UNDERSCORE
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
#else
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
#endif
    /* a..o */
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    /* p..z */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
#ifdef LEX_ID_WITH_UTF8
    /* utf-8 */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#else
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#endif
};

#define lex_isalnum(c) (lex_alnum[(unsigned char)(c)])
#endif

#ifndef lex_isbindigit
#define lex_isbindigit(c) ((c) == '0' || (c) == '1')
#endif

#ifndef lex_isoctdigit
#define lex_isoctdigit(c) ((unsigned)((c) - '0') < 8)
#endif

#ifndef lex_isdigit
#define lex_isdigit(c) ((unsigned)(c) >= '0' && (unsigned)(c) <= '9')
#endif

#ifndef lex_ishexdigit
#define lex_ishexdigit(c) (((c) >= '0' && ((unsigned)c) <= '9') || ((unsigned)(c | 0x20) >= 'a' && (unsigned)(c | 0x20) <= 'f'))
#endif

#ifndef lex_isctrl
#include <ctype.h>
#define lex_isctrl(c) (((unsigned)c) < 0x20 || (c) == 0x7f)
#endif

#ifndef lex_isblank
#define lex_isblank(c) ((c) == ' ' || (c) == '\t')
#endif

#ifndef lex_iszterm
#define lex_iszterm(c) ((c) == '\0')
#endif

/*
 * If ZTERM is disabled, zero will be a LEX_CTRL token
 * and allowed to be embedded in comments and strings, or
 * elsewhere, as long as the parser accepts the token.
 */
#ifdef LEX_DISABLE_ZTERM
#undef lex_iszterm
#define lex_iszterm(c) (0)
#endif

/*
 * The mode is normally LEX_MODE_NORMAL = 0 initially, or the returned
 * mode from a previous call, unless LEX_MODE_INVALID = 1 was returned.
 * If a buffer stopped in the middle of a string or a comment, the mode
 * will reflect that. In all cases some amount of recovery is needed
 * before starting a new buffer - see detailed comments in header file.
 * If only a single buffer is used, special handling is still needed if
 * the last line contains a single line comment because it will not be
 * terminated, but it amounts to replace the emitted unterminated
 * comment token with an end of comment token.
 *
 * Instead of 0, the mode can initially also be LEX_MODE_BOM - it will
 * an strip optional BOM before moving to normal mode. Currently only
 * UTF-8 BOM is supported, and this is unlikely to change.
 *
 * The context variable is user-defined and available to emitter macros.
 * It may be null if unused.
 *
 */
static int lex(const char *buf, size_t len, int mode, void *context)
{
    const char *p, *q, *s, *d;
#if 0
    /* TODO: old, remove this */
    , *z, *f;
#endif

    p = buf;        /* next char */
    q = p + len;    /* end of buffer */
    s = p;          /* start of token */
    d = p;          /* end of integer part */

#if 0
    /* TODO: old, remove this */

    /* Used for float and leading zero detection in numerics. */
    z = p;
    f = p;
#endif

    /*
     * Handle mid string and mid comment for reentering across
     * buffer boundaries. Strip embedded counter from mode.
     */
    switch(mode & (LEX_MODE_COUNT_BASE - 1)) {

    case LEX_MODE_NORMAL:
        goto lex_mode_normal;

    case LEX_MODE_BOM:
        goto lex_mode_bom;

#ifdef LEX_C_STRING
    case LEX_MODE_C_STRING:
        goto lex_mode_c_string;
#endif
#ifdef LEX_PYTHON_BLOCK_STRING
    case LEX_MODE_PYTHON_BLOCK_STRING:
        goto lex_mode_python_block_string;
#endif
#ifdef LEX_C_STRING_SQ
    case LEX_MODE_C_STRING_SQ:
        goto lex_mode_c_string_sq;
#endif
#ifdef LEX_PYTHON_BLOCK_STRING_SQ
    case LEX_MODE_PYTHON_BLOCK_STRING_SQ:
        goto lex_mode_python_block_string_sq;
#endif
#ifdef LEX_C_BLOCK_COMMENT
    case LEX_MODE_C_BLOCK_COMMENT:
        goto lex_mode_c_block_comment;
#endif
#if defined(LEX_SHELL_LINE_COMMENT) || defined(LEX_C99_LINE_COMMENT)
    case LEX_MODE_LINE_COMMENT:
        goto lex_mode_line_comment;
#endif
#ifdef LEX_JULIA_NESTED_COMMENT
    case LEX_MODE_JULIA_NESTED_COMMENT:
        goto lex_mode_julia_nested_comment;
#endif

    default:
        /*
         * This is mostly to kill unused label warning when comments
         * are disabled.
         */
        goto lex_mode_exit;
    }

lex_mode_bom:

    mode = LEX_MODE_BOM;

    /*
     * Special entry mode to consume utf-8 bom if present. We don't
     * support other boms, but we would use the same token if we did.
     *
     * We generally expect no bom present, but it is here if needed
     * without requiring ugly hacks elsewhere.
     */
    if (p + 3 < q && p[0] == '\xef' && p[1] == '\xbb' && p[2] == '\xbf') {
        p += 3;
        lex_emit_bom(s, p);
    }
    goto lex_mode_normal;

/* If source is updated, also update LEX_C_STRING_SQ accordingly. */
#ifdef LEX_C_STRING
lex_mode_c_string:

    mode = LEX_MODE_C_STRING;

    for (;;) {
        --p;
        /* We do not allow blanks that are also control characters, such as \t. */
        while (++p != q && *p != '\\' && *p != '\"' && !lex_isctrl(*p)) {
        }
        if (s != p) {
            lex_emit_string_part(s, p);
            s = p;
        }
        if (*p == '\"') {
            ++p;
            lex_emit_string_end(s, p);
            goto lex_mode_normal;
        }
        if (p == q || lex_iszterm(*p)) {
            lex_emit_string_unterminated(p);
            goto lex_mode_normal;
        }
        if (*p == '\\') {
            ++p;
             /* Escape is only itself, whatever is escped follows separately. */
            lex_emit_string_escape(s, p);
            s = p;
            if (p == q || lex_iszterm(*p)) {
                lex_emit_string_unterminated(p);
                goto lex_mode_normal;
            }
            if (*p == '\\' || *p == '\"') {
                ++p;
                continue;
            }
            /*
             * Flag only relevant for single line strings, as it
             * controls whether we fail on unterminated string at line
             * ending with '\'.
             *
             * Julia does not support line continuation in strings
             * (or elsewhere). C, Python, and Javascript do.
             */
#ifndef LEX_DISABLE_STRING_CONT
            if (*p == '\n') {
                if (++p != q && *p == '\r') {
                    ++p;
                }
                lex_emit_string_newline(s, p);
                s = p;
                continue;
            }
            if (*p == '\r') {
                if (++p != q && *p == '\n') {
                    ++p;
                }
                lex_emit_string_newline(s, p);
                s = p;
                continue;
            }
#endif
        }
        if (*p == '\n' || *p == '\r') {
            lex_emit_string_unterminated(p);
            goto lex_mode_normal;
        }
        ++p;
        lex_emit_string_ctrl(s);
        s = p;
    }
#endif

/*
 * This is a copy if LEX_C_STRING with single quote. It's not DRY, but
 * no reason to parameterized inner loops, just because. Recopy of
 * changes are to the above.
 *
 * Even if single quote is only used for CHAR types, it makes sense to
 * parse as a full string since there can be all sorts of unicocde
 * escapes and line continuations, newlines to report and unexpected
 * control characters to deal with.
 */
#ifdef LEX_C_STRING_SQ
lex_mode_c_string_sq:

    mode = LEX_MODE_C_STRING_SQ;

    for (;;) {
        --p;
        while (++p != q && *p != '\\' && *p != '\'' && !lex_isctrl(*p)) {
        }
        if (s != p) {
            lex_emit_string_part(s, p);
            s = p;
        }
        if (*p == '\'') {
            ++p;
            lex_emit_string_end(s, p);
            goto lex_mode_normal;
        }
        if (p == q || lex_iszterm(*p)) {
            lex_emit_string_unterminated(p);
            goto lex_mode_normal;
        }
        if (*p == '\\') {
            ++p;
             /* Escape is only itself, whatever is escped follows separately. */
            lex_emit_string_escape(s, p);
            s = p;
            if (p == q || lex_iszterm(*p)) {
                lex_emit_string_unterminated(p);
                goto lex_mode_normal;
            }
            if (*p == '\\' || *p == '\'') {
                ++p;
                continue;
            }
            /*
             * Flag only relevant for single line strings, as it
             * controls whether we fail on unterminated string at line
             * ending with '\'.
             *
             * Julia does not support line continuation in strings
             * (or elsewhere). C, Python, and Javascript do.
             */
#ifndef LEX_DISABLE_STRING_CONT
            if (*p == '\n') {
                if (++p != q && *p == '\r') {
                    ++p;
                }
                lex_emit_string_newline(s, p);
                s = p;
                continue;
            }
            if (*p == '\r') {
                if (++p != q && *p == '\n') {
                    ++p;
                }
                lex_emit_string_newline(s, p);
                s = p;
                continue;
            }
#endif
        }
        if (*p == '\n' || *p == '\r') {
            lex_emit_string_unterminated(p);
            goto lex_mode_normal;
        }
        ++p;
        lex_emit_string_ctrl(s);
        s = p;
    }
#endif

/*
 * """ Triple quoted Python block strings. """
 * Single quoted version (''') is a direct copy, update both places
 * if a changed is needed.
 *
 * Note: there is no point in disabling line continuation
 * for block strings, since it only affects unterminated
 * string errors at newline. It all comes down to how
 * escaped newline is interpreted by the parser.
 */
#ifdef LEX_PYTHON_BLOCK_STRING
lex_mode_python_block_string:

    mode = LEX_MODE_PYTHON_BLOCK_STRING;

    for (;;) {
        --p;
        while (++p != q && *p != '\\' && !lex_isctrl(*p)) {
            if (*p == '\"' && p + 2 < q && p[1] == '\"' && p[2] == '\"') {
                break;
            }
        }
        if (s != p) {
            lex_emit_string_part(s, p);
            s = p;
        }
        if (p == q || lex_iszterm(*p)) {
            lex_emit_string_unterminated(p);
            goto lex_mode_normal;
        }
        if (*p == '\"') {
            p += 3;
            lex_emit_string_end(s, p);
            goto lex_mode_normal;
        }
        if (*p == '\\') {
             /* Escape is only itself, allowing parser to interpret and validate. */
            ++p;
            lex_emit_string_escape(s, p);
            s = p;
            if (p + 1 != q && (*p == '\\' || *p == '\"')) {
                ++p;
            }
            continue;
        }
        if (*p == '\n') {
            if (++p != q && *p == '\r') {
                ++p;
            }
            lex_emit_string_newline(s, p);
            s = p;
            continue;
        }
        if (*p == '\r') {
            if (++p != q && *p == '\n') {
                ++p;
            }
            lex_emit_string_newline(s, p);
            s = p;
            continue;
        }
        ++p;
        lex_emit_string_ctrl(s);
        s = p;
    }
#endif

/*
 * Python ''' style strings.
 * Direct copy of """ quote version, update both if changed.
 */
#ifdef LEX_PYTHON_BLOCK_STRING_SQ
lex_mode_python_block_string_sq:

    mode = LEX_MODE_PYTHON_BLOCK_STRING_SQ;

    for (;;) {
        --p;
        while (++p != q && *p != '\\' && !lex_isctrl(*p)) {
            if (*p == '\'' && p + 2 < q && p[1] == '\'' && p[2] == '\'') {
                break;
            }
        }
        if (s != p) {
            lex_emit_string_part(s, p);
            s = p;
        }
        if (p == q || lex_iszterm(*p)) {
            lex_emit_string_unterminated(p);
            goto lex_mode_normal;
        }
        if (*p == '\'') {
            p += 3;
            lex_emit_string_end(s, p);
            goto lex_mode_normal;
        }
        if (*p == '\\') {
             /* Escape is only itself, allowing parser to interpret and validate. */
            ++p;
            lex_emit_string_escape(s, p);
            s = p;
            if (p + 1 != q && (*p == '\\' || *p == '\'')) {
                ++p;
            }
            continue;
        }
        if (*p == '\n') {
            if (++p != q && *p == '\r') {
                ++p;
            }
            lex_emit_string_newline(s, p);
            s = p;
            continue;
        }
        if (*p == '\r') {
            if (++p != q && *p == '\n') {
                ++p;
            }
            lex_emit_string_newline(s, p);
            s = p;
            continue;
        }
        ++p;
        lex_emit_string_ctrl(s);
        s = p;
    }
#endif

/*
 * We don't really care if it is a shell style comment or a C99,
 * or any other line oriented commment, as the termination is
 * the same.
 */
#if defined(LEX_SHELL_LINE_COMMENT) || defined(LEX_C99_LINE_COMMENT)
lex_mode_line_comment:

    mode = LEX_MODE_LINE_COMMENT;

    for (;;) {
        --p;
        while (++p != q && (!lex_isctrl(*p))) {
        }
        if (s != p) {
            lex_emit_comment_part(s, p);
            s = p;
        }
        if (p == q || lex_iszterm(*p)) {
            /*
             * Unterminated comment here is not necessarily true,
             * not even likely, nor possible, but we do this to
             * handle buffer switch consistently: any non-normal
             * mode exit will have an unterminated token to fix up.
             * Here it would be conversion to end of comment, which
             * we cannot know yet, since the line might continue in
             * the next buffer. This is a zero length token.
             */
            lex_emit_comment_unterminated(p);
            goto lex_mode_exit;
        }
        if (*p == '\n' || *p == '\r') {
            lex_emit_comment_end(s, p);
            goto lex_mode_normal;
        }
        ++p;
        lex_emit_comment_ctrl(s);
        s = p;
    }
#endif

#ifdef LEX_C_BLOCK_COMMENT
lex_mode_c_block_comment:

    mode = LEX_MODE_C_BLOCK_COMMENT;

    for (;;) {
        --p;
        while (++p != q && (!lex_isctrl(*p))) {
            if (*p == '/' && p[-1] == '*') {
                --p;
                break;
            }
        }
        if (s != p) {
            lex_emit_comment_part(s, p);
            s = p;
        }
        if (p == q || lex_iszterm(*p)) {
            lex_emit_comment_unterminated(p);
            goto lex_mode_exit;
        }
        if (*p == '\n') {
            if (++p != q && *p == '\r') {
                ++p;
            }
            lex_emit_newline(s, p);
            s = p;
            continue;
        }
        if (*p == '\r') {
            if (++p != q && *p == '\n') {
                ++p;
            }
            lex_emit_newline(s, p);
            s = p;
            continue;
        }
        if (lex_isctrl(*p)) {
            ++p;
            lex_emit_comment_ctrl(s);
            s = p;
            continue;
        }
        p += 2;
        lex_emit_comment_end(s, p);
        s = p;
        goto lex_mode_normal;
    }
#endif

    /* Julia nests block comments as #= ... #= ...=# ... =# across multiple lines. */
#ifdef LEX_JULIA_NESTED_COMMENT
lex_mode_julia_nested_comment:

    /* Preserve nesting level on re-entrance. */
    if ((mode & (LEX_MODE_COUNT_BASE - 1)) != LEX_MODE_JULIA_NESTED_COMMENT) {
        mode = LEX_MODE_JULIA_NESTED_COMMENT;
    }
    /* We have already entered. */
    mode += LEX_MODE_COUNT_BASE;

    for (;;) {
        --p;
        while (++p != q && !lex_isctrl(*p)) {
            if (*p == '#') {
                if (p[-1] == '=') {
                    --p;
                    break;
                }
                if (p + 1 != q && p[1] == '=') {
                    break;
                }
            }
        }
        if (s != p) {
            lex_emit_comment_part(s, p);
            s = p;
        }
        if (p == q || lex_iszterm(*p)) {
            lex_emit_comment_unterminated(p);
            goto lex_mode_exit;
        }
        if (*p == '\n') {
            if (++p != q && *p == '\r') {
                ++p;
            }
            lex_emit_newline(s, p);
            s = p;
            continue;
        }
        if (*p == '\r') {
            if (++p != q && *p == '\n') {
                ++p;
            }
            lex_emit_newline(s, p);
            s = p;
            continue;
        }
        if (lex_isctrl(*p)) {
            ++p;
            lex_emit_comment_ctrl(s);
            s = p;
            continue;
        }
        if (*p == '=') {
            p += 2;
            lex_emit_comment_end(s, p);
            s = p;
            mode -= LEX_MODE_COUNT_BASE;
            if (mode / LEX_MODE_COUNT_BASE > 0) {
                continue;
            }
            goto lex_mode_normal;
        }
        /* The upper bits are used as counter. */
        mode += LEX_MODE_COUNT_BASE;
        p += 2;
        lex_emit_comment_begin(s, p, 0);
        s = p;
        if (mode / LEX_MODE_COUNT_BASE > LEX_MAX_NESTING_LEVELS) {
            /* Prevent malicious input from overflowing counter. */
            lex_emit_comment_deeply_nested(p);
            lex_emit_abort(p);
            return mode;
        }
    }
#endif

/* Unlike other modes, we can always jump here without updating token start `s` first. */
lex_mode_normal:

    mode = LEX_MODE_NORMAL;

    while (p != q) {
        s = p;

        switch(*p) {

#ifndef LEX_DISABLE_ZTERM
        case '\0':
            lex_emit_eos(s, p);
            return mode;
#endif

        /* \v, \f etc. are covered by the CTRL token, don't put it here. */
        case '\t': case ' ':
            while (++p != q && lex_isblank(*p)) {
            }
            lex_emit_blank(s, p);
            continue;

        /*
         * Newline should be emitter in all constructs, also comments
         * and strings which have their own newline handling.
         * Only one line is emitted at a time permitting simple line
         * counting.
         */
        case '\n':
            if (++p != q && *p == '\r') {
                ++p;
            }
            lex_emit_newline(s, p);
            continue;

        case '\r':
            if (++p != q && *p == '\n') {
                ++p;
            }
            lex_emit_newline(s, p);
            continue;

            /*
             * C-style string, and Python style triple double quote
             * delimited multi-line string. Prefix and suffix symbols
             * should be parsed separately, e.g. L"hello" are two
             * tokens.
             */
#if defined(LEX_C_STRING) || defined(LEX_PYTHON_BLOCK_STRING)
        case '\"':
#ifdef LEX_PYTHON_BLOCK_STRING
            if (p + 2 < q && p[1] == '\"' && p[2] == '\"') {
                p += 3;
                lex_emit_string_begin(s, p);
                s = p;
                goto lex_mode_python_block_string;
            }
#endif
#ifdef LEX_C_STRING
            ++p;
            lex_emit_string_begin(s, p);
            s = p;
            goto lex_mode_c_string;
#endif
#endif

            /*
             * Single quoted version of strings, otherwise identical
             * behavior. Can also be used for char constants if checked
             * by parser subsequently.
             */
#if defined(LEX_C_STRING_SQ) || defined(LEX_PYTHON_BLOCK_STRING_SQ)
        case '\'':
#ifdef LEX_PYTHON_BLOCK_STRING_SQ
            if (p + 2 < q && p[1] == '\'' && p[2] == '\'') {
                p += 3;
                lex_emit_string_begin(s, p);
                s = p;
                goto lex_mode_python_block_string_sq;
            }
#endif
#ifdef LEX_C_STRING_SQ
            ++p;
            lex_emit_string_begin(s, p);
            s = p;
            goto lex_mode_c_string_sq;
#endif
#endif

#if defined(LEX_SHELL_LINE_COMMENT) || defined(LEX_JULIA_NESTED_COMMENT)
            /*
             * Line comment excluding terminal line break.
             *
             * See also C99 line comment `//`.
             *
             * Julia uses `#=` and `=#` for nested block comments.
             * (According to Julia developers, '#=` is motivated by `=`
             * not being likely to start anything that you would put a
             * comment around, unlike `#{`, `}#` or `#(`, `)#`)).
             *
             * Some known doc comment formats are identified and
             * included in the comment_begin token.
             */
        case '#':
            ++p;
#ifdef LEX_JULIA_NESTED_COMMENT
            if (p != q && *p == '=') {
                ++p;
                lex_emit_comment_begin(s, p, 0);
                s = p;
                goto lex_mode_julia_nested_comment;
            }
#endif
            lex_emit_comment_begin(s, p, 0);
            s = p;
            goto lex_mode_line_comment;
#endif

        case '/':
            ++p;
            if (p != q) {
                switch (*p) {
#ifdef LEX_C99_LINE_COMMENT
                case '/':
                    ++p;
                    p += p != q && (*p == '/' || *p == '!');
                    lex_emit_comment_begin(s, p, (p - s == 3));
                    s = p;
                    goto lex_mode_line_comment;
#endif
#ifdef LEX_C_BLOCK_COMMENT
                case '*':
                    ++p;
                    p += p != q && (*p == '*' || *p == '!');
                    lex_emit_comment_begin(s, p, (p - s == 3));
                    s = p;
                    goto lex_mode_c_block_comment;
#endif
                case '=':
                    ++p;
                    lex_emit_compound_op('/', '=', s, p);
                    continue;
                default:
                    break;
                }
            }
            lex_emit_op('/', s, p);
            continue;

        case '(': case ')': case '[': case ']': case '{': case '}':
        case ',': case ';': case '\\': case '?':
            ++p;
            lex_emit_op(*s, s, p);
            continue;

        case '%': case '!': case '~': case '^':
            ++p;
            if (p != q && *p == '=') {
                ++p;
                lex_emit_compound_op(*s, '=', s, p);
                continue;
            }
            lex_emit_op(*s, s, p);
            continue;

        case '|':
            ++p;
            if (p != q) {
                switch (*p) {
                case '=':
                    ++p;
                    lex_emit_compound_op('|', '=', s, p);
                    continue;
                case '|':
                    ++p;
                    lex_emit_compound_op('|', '|', s, p);
                    break;
                default:
                    break;
                }
            }
            lex_emit_op('|', s, p);
            continue;

        case '&':
            ++p;
            if (p != q) {
                switch (*p) {
                case '=':
                    ++p;
                    lex_emit_compound_op('&', '=', s, p);
                    continue;
                case '&':
                    ++p;
                    lex_emit_compound_op('&', '&', s, p);
                    break;
                default:
                    break;
                }
            }
            lex_emit_op('&', s, p);
            continue;

        case '=':
            ++p;
            if (p != q) {
                switch (*p) {
                case '>':
                    ++p;
                    lex_emit_compound_op('=', '>', s, p);
                    continue;
                case '=':
                    ++p;
                    if (p != q && *p == '=') {
                        ++p;
                        lex_emit_tricompound_op('=', '=', '=', s, p);
                        continue;
                    }
                    lex_emit_compound_op('=', '=', s, p);
                    break;
                default:
                    break;
                }
            }
            lex_emit_op('=', s, p);
            continue;

        case ':':
            ++p;
            if (p != q) {
                switch (*p) {
                case '=':
                    ++p;
                    lex_emit_compound_op(':', '=', s, p);
                    continue;
                case ':':
                    ++p;
                    if (p != q && *p == '=') {
                        ++p;
                        lex_emit_tricompound_op(':', ':', '=', s, p);
                        continue;
                    }
                    lex_emit_compound_op(':', ':', s, p);
                    continue;
                default:
                    break;
                }
            }
            lex_emit_op(':', s, p);
            continue;

        case '*':
            ++p;
            if (p != q) {
                switch (*p) {
                case '=':
                    lex_emit_compound_op('*', '=', s, p);
                    continue;
                case '*':
                    /* **= hardly used anywhere? */
                    lex_emit_compound_op('*', '*', s, p);
                    continue;
                default:
                    break;
                }
            }
            lex_emit_op('*', s, p);
            continue;

        case '<':
            ++p;
            if (p != q) {
                switch (*p) {
                case '-':
                    ++p;
                    lex_emit_compound_op('<', '-', s, p);
                    continue;
                case '=':
                    ++p;
                    lex_emit_compound_op('<', '=', s, p);
                    continue;
                case '<':
                    ++p;
                    if (p != q) {
                        switch (*p) {
                        case '=':
                            ++p;
                            lex_emit_tricompound_op('<', '<', '=', s, p);
                            continue;
                        case '<':
                            ++p;
                            if (p != q && *p == '=') {
                                ++p;
                                lex_emit_quadcompound_op('<', '<', '<', '=', s, p);
                                continue;
                            }
                            lex_emit_tricompound_op('<', '<', '<', s, p);
                            continue;
                        default:
                            break;
                        }
                    }
                    lex_emit_compound_op('<', '<', s, p);
                    continue;
                default:
                    break;
                }
            }
            lex_emit_op('<', s, p);
            continue;

        case '>':
            ++p;
            if (p != q) {
                switch (*p) {
                case '=':
                    ++p;
                    lex_emit_compound_op('>', '=', s, p);
                    continue;
                case '>':
                    ++p;
                    if (p != q) {
                        switch (*p) {
                        case '=':
                            ++p;
                            lex_emit_tricompound_op('>', '>', '=', s, p);
                            continue;
                        case '>':
                            ++p;
                            if (p != q && *p == '=') {
                                ++p;
                                lex_emit_quadcompound_op('>', '>', '>', '=', s, p);
                                continue;
                            }
                            lex_emit_tricompound_op('>', '>', '>', s, p);
                            continue;
                        default:
                            break;
                        }
                    }
                    lex_emit_compound_op('>', '>', s, p);
                    continue;
                default:
                    break;
                }
            }
            lex_emit_op('>', s, p);
            continue;

        case '-':
            ++p;
            if (p != q) {
                switch (*p) {
                case '=':
                    ++p;
                    lex_emit_compound_op('-', '=', s, p);
                    continue;
                case '-':
                    ++p;
                    lex_emit_compound_op('-', '-', s, p);
                    continue;
                case '>':
                    ++p;
                    lex_emit_compound_op('-', '>', s, p);
                    continue;
                default:
                    break;
                }
            }
            lex_emit_op('-', s, p);
            continue;

        case '+':
            ++p;
            if (p != q) {
                switch (*p) {
                case '=':
                    ++p;
                    lex_emit_compound_op('+', '=', s, p);
                    continue;

                case '+':
                    ++p;
                    lex_emit_compound_op('+', '+', s, p);
                    continue;
                default:
                    break;
                }
            }
            lex_emit_op('+', s, p);
            continue;

        case '.':
            ++p;
            if (p != q) {
                switch (*p) {
                case '0': case '1': case '2': case '3': case '4':
                case '5': case '6': case '7': case '8': case '9':
                    d = s;
                    goto lex_dot_to_fraction_part;
                case '.':
                    ++p;
                    if (p != q && *p == '.') {
                        ++p;
                        lex_emit_tricompound_op('.', '.', '.', s, p);
                        continue;
                    }
                    lex_emit_compound_op('.', '.', s, p);
                    continue;
                default:
                    break;
                }
            }
            lex_emit_op('.', s, p);
            continue;

        case '0':
            if (++p != q) {
                switch (*p) {
#ifdef LEX_C_OCTAL_NUMERIC

                case '0': case '1': case '2': case '3':
                case '4': case '5': case '6': case '7':
                    while (++p != q && lex_isoctdigit(*p)) {
                    }
                    d = p;
                    if (p != q) {
                        /*
                         * Leading zeroes like 00.10 are valid C
                         * floating point constants.
                         */
                        if (*p == '.') {
                            goto lex_c_octal_to_fraction_part;
                        }
                        if (*p == 'e' || *p == 'E') {
                            goto lex_c_octal_to_exponent_part;
                        }
                    }
                    lex_emit_octal(s, p);
                    /*
                     * If we have a number like 0079, it becomes
                     * 007(octal), 9(decimal). The parser should
                     * deal with this.
                     *
                     * To add to confusion i64 is a C integer suffix
                     * like in 007i64, but 2+2i is a Go complex
                     * constant. (Not specific to octals).
                     *
                     * This can all be handled by having the parser inspect
                     * following identifier or numeric, parser
                     * here meaning a lexer post processing step, not
                     * necessarily the parser itself.
                     */

                    continue;
#else
                    /*
                     * All integers reach default and enter
                     * integer part. As a result, leading zeroes are
                     * mapped to floats and integers which matches
                     * Julia behavior. Other languages should decide
                     * if leading zero is valid or not. JSON
                     * disallows leading zero.
                     */
#endif

#ifdef LEX_JULIA_OCTAL_NUMERIC
                    /*
                     * This is the style of octal, not 100% Julia
                     * compatible. Also define Julia numeric to enforce
                     * lower case.
                     */
#ifndef LEX_LOWER_CASE_NUMERIC_PREFIX
                    /* See also hex 0X. Julia v.0.3 uses lower case only here. */
                case 'O':
#endif
                    /*
                     * Julia accepts 0o700 as octal and 0b100 as
                     * binary, and 0xa00 as hex, and 0100 as
                     * integer, and 1e2 as 64 bit float and 1f2 as
                     * 32 bit float. Julia 0.3 does not support
                     * octal and binary fractions.
                     */
                case 'o':
                    while (++p != q && lex_isoctdigit(*p)) {
                    }
                    lex_emit_octal(s, p);
                    /* Avoid hitting int fall through. */
                    continue;
#endif
#ifdef LEX_BINARY_NUMERIC
                    /* Binary in C++14. */
                case 'b':
#ifndef LEX_LOWER_CASE_NUMERIC_PREFIX
                    /* See also hex 0X. Julia v.0.3 uses lower case only here. */
                case 'B':
#endif
                    while (++p != q && lex_isbindigit(*p)) {
                    }
                    lex_emit_binary(s, p);
                    /* Avoid hitting int fall through. */
                    continue;
#endif
#ifdef LEX_HEX_NUMERIC
                case 'x':
#ifndef LEX_LOWER_CASE_NUMERIC_PREFIX
                    /*
                     * Julia v0.3 does not allow this, it thinks 0X1 is
                     * 0 * X1, X1 being an identifier.
                     * while 0x1 is a hex value due to precedence.
                     *
                     * TODO: This might change.
                     */

                case 'X':
#endif
                    while (++p != q && lex_ishexdigit(*p)) {
                    }
#ifdef LEX_HEX_FLOAT_NUMERIC
                    /*
                     * Most hexadecimal floating poing conversion
                     * functions, including Pythons
                     * float.fromhex("0x1.0"), Julias parse
                     * function, and and C strtod on
                     * supporting platforms, will parse without
                     * exponent. The same languages do not support
                     * literal constants without the p exponent.
                     * First it is named p because e is a hex digit,
                     * second, the float suffix f is also a hex
                     * digit: 0x1.f is ambigious in C without that
                     * rule. Conversions have no such ambiguity.
                     * In Julia, juxtaposition means that 0x1.f
                     * could mean 0x1p0 * f or 0x1.fp0.
                     *
                     * Since we are not doing conversion here but
                     * lexing a stream, we opt to require the p
                     * suffix because making it optional could end
                     * up consuming parts of the next token.
                     *
                     * But, we also make a flag to make the exponent
                     * optional, anyway. It could be used for better
                     * error reporting than just consuming the hex
                     * part since we likely should accept the ambigous
                     * syntax either way.
                     */
                    d = p;
                    if (p != q && *p == '.') {
                        while (++p != q && lex_ishexdigit(*p)) {
                        }
                    }
                    if (p != q && (*p == 'p' || *p == 'P')) {
                        if (++p != q && *p != '+' && *p != '-') {
                            --p;
                        }
                        /* The exponent is a decimal power of 2. */
                        while (++p != q && lex_isdigit(*p)) {
                        }
                        lex_emit_hex_float(s, p);
                        continue;
                    }
#ifdef LEX_HEX_FLOAT_OPTIONAL_EXPONENT
                    if (d != p) {
                        lex_emit_hex_float(s, p);
                        continue;
                    }
#else
                    /*
                     * Backtrack to decimal point. We require p to
                     * be present because we could otherwise consume
                     * part of the next token.
                     */
                    p = d;
#endif
#endif /* LEX_HEX_FLOAT_NUMERIC */
                    lex_emit_hex(s, p);
                    continue;
#endif /* LEX_HEX_NUMERIC */

                default:
                    /*
                     * This means leading zeroes like 001 or 001.0 are
                     * treated like like int and float respectively,
                     * iff C octals are flaggged out. Otherwise they
                     * become 001(octal), and 001(octal),.0(float)
                     * which should be treated as an error because
                     * future extensions might allow octal floats.
                     * (Not likely, but interpretation is ambigious).
                     */
                    break;
                } /* Switch under '0' case. */

                /*
                 * Pure single digit '0' is an octal number in the C
                 * spec. We have the option to treat it as an integer,
                 * or as an octal. For strict C behavior, this can be
                 * flagged in, but is disabled by default. It only
                 * applies to single digit 0. Thus, with C octal
                 * enabled, leading zeroes always go octal.
                 */
            } /* If condition around switch under '0' case. */
            --p;
            goto lex_fallthrough_1; /* silence warning */

        lex_fallthrough_1:
            /* Leading integer digit in C integers. */
        case '1': case '2': case '3': case '4': case '5':
        case '6': case '7': case '8': case '9':
            while (++p && lex_isdigit(*p)) {
            }
            d = p;
            if (*p == '.') {
/* Silence unused label warnings when features are disabled. */
#ifdef LEX_C_OCTAL_NUMERIC
lex_c_octal_to_fraction_part:
#endif
lex_dot_to_fraction_part:
                while (++p != q && lex_isdigit(*p)) {
                }
            }
            if (p != q && (*p == 'e' || *p == 'E')) {
/* Silence unused label warnings when features are disabled. */
#ifdef LEX_C_OCTAL_NUMERIC
lex_c_octal_to_exponent_part:
#endif
                if (++p != q && *p != '+' && *p != '-') {
                    --p;
                }
                while (++p != q && lex_isdigit(*p)) {
                }
            }
            if (d != p) {
                lex_emit_float(s, p);
            } else {
#ifdef LEX_C_OCTAL_NUMERIC
                if (*s == '0') {
                    lex_emit_octal(s, p);
                    continue;
                }
#endif
                lex_emit_int(s, p);
            }
            continue;

#ifndef LEX_ID_WITHOUT_UNDERSCORE
            case '_':
#endif
            case 'A': case 'B': case 'C': case 'D': case 'E':
            case 'F': case 'G': case 'H': case 'I': case 'J':
            case 'K': case 'L': case 'M': case 'N': case 'O':
            case 'P': case 'Q': case 'R': case 'S': case 'T':
            case 'U': case 'V': case 'W': case 'X': case 'Y':
            case 'Z':
            case 'a': case 'b': case 'c': case 'd': case 'e':
            case 'f': case 'g': case 'h': case 'i': case 'j':
            case 'k': case 'l': case 'm': case 'n': case 'o':
            case 'p': case 'q': case 'r': case 's': case 't':
            case 'u': case 'v': case 'w': case 'x': case 'y':
            case 'z':

                /*
                 * We do not try to ensure utf-8 is terminated correctly nor
                 * that any unicode character above ASCII is a character
                 * suitable for identifiers.
                 *
                 * tag is calculated for keyword lookup, and we assume these
                 * are always ASCII-7bit.  It has the form: length, first
                 * char, second, char, last char in lsb to msb order. If the
                 * second char is missing, it becomes '\0'. The tag is not
                 * entirely unique, but suitable for fast lookup.
                 *
                 * If utf-8 appears in tag, the tag is undefined except the
                 * length is valid or overflows (meaning longer than any
                 * keyword and thus safe to compare against if tag matches).
                 *
                 * If the grammar is case insensitive, the tag be can
                 * downcased trivially by or'ring with 0x20202000 which
                 * preserves the length field (clever design by ASCII
                 * designers). After tag matching, a case insentive
                 * compare is obviously also needed against the full lexeme.
                 */

                {
                    unsigned long tag;

                    tag = (unsigned long)*p << 8;
                    if (++p != q && lex_isalnum(*p)) {
                        tag |= (unsigned long)*p << 16;
                        while (++p != q && lex_isalnum(*p)) {
                        }
                    }
                    tag |= (unsigned long)p[-1] << 24;
                    tag |= (unsigned char)(p - s) + (unsigned long)'0';
                    lex_emit_id(s, p, tag);
                    continue;
                }

            default:

#ifdef LEX_ID_WITH_UTF8
                /*
                 * Identifier again, in case it starts with a utf-8 lead
                 * character. This time we can ignore the tag, except the
                 * length char must be valid to avoid buffer overruns
                 * on potential kw check upstream.
                 */
                if (*p & '\x80') {
                    unsigned long tag;

                    while (++p != q && lex_isalnum(*p)) {
                    }
                    tag = (unsigned char)(p - s) + '0';
                    lex_emit_id(s, p, tag);
                    continue;
                }
#endif
                ++p;
                /* normally 0x7f DEL and 0x00..0x1f incl. */
                if (lex_isctrl(*s) && !lex_isblank(*s)) {
                    lex_emit_ctrl(s);
                } else {
                    lex_emit_symbol(*s, s, p);
                }
                continue;
        } /* Main switch in normal mode. */
    } /* Main while loop in normal mode. */

lex_mode_exit:
    if (mode == LEX_MODE_INVALID) {
        return mode;
    }

#ifndef LEX_DISABLE_ZTERM
    if (p != q && lex_iszterm(*p)) {
        lex_emit_eos(s, p);
        return mode;
    }
#endif
    lex_emit_eob(p);
    return mode;
}

