#ifndef LEX_TOKENS_H
#define LEX_TOKENS_H

/* Define LEX_DEBUG to enable token printing and describing functions. */


enum {

    /*
     * EOF is not emitted by lexer, but may be used by driver after
     * last buffer is processed.
     */
    LEX_TOK_EOF = 0,

    /*
     * Either EOB or EOS is emitted as the last token before exit,
     * or also ABORT in some lexers. Unterminated string or comment
     * will be emitted immediately one of these when relevant.
     *
     * It may be useful to redefine lex_emit_eos and lex_emit_eob to
     * produce LEX_TOK_EOF or error directly for simple string lexing.
     */
    LEX_TOK_EOB = 1,
    LEX_TOK_EOS = 2,

    /*
     * ABORT can be used for early exit by some lexers while other
     * lexers may choose to run to buffer end regardless of input (with
     * the exception of deeply nested comments).
     */
    LEX_TOK_ABORT = 3,

    /*
     * Byte order marker. Only happen if lexer was started in bom mode
     * and the input stream contains a leading bom marker.
     * The token can only be the first token in the stream. Utf-8 is the
     * only supported bom, but the lexeme may be checked in case other
     * boms are added later. Normally it is routed to lex_emit_other
     * along with comments so it just ignores the bom if present. It is
     * generally recommended to consume utf-8 bom for interoperability,
     * but also to not store it for the same reason.
     */
    LEX_TOK_BOM,

    /*
     * Any control character that is not newline or blank will be
     * emitted as single character token here. This token is discussed
     * in several comments below. For strings and comments, also
     * blank control characters will be emitted since they are usually
     * not desired unexpectd.
     */
    LEX_TOK_CTRL,
    LEX_TOK_STRING_CTRL,
    LEX_TOK_COMMENT_CTRL,

    /*
     * Any printable ASCII character that is not otherwise consumed will
     * be issued as a single length symbol token. Further discussion
     * below. The symbol and CTRL tokens ensure that the entire input
     * stream is covered by tokens. If utf-8 identifies have not been
     * flagged, utf-8 leading characters may also end up here, and so
     * my utf-8 characters in general, that are not viewed as valid
     * identifiers (depending on configuration).
     */
    LEX_TOK_SYMBOL,

    /*
     * Variable length identifier starting with (_A-Za-z) by default and
     * followed by zero or more (_A-Za-z0-9) characters. (_) can be
     * flagged out. utf-8 can be flagged in. Be default any non-ASCII
     * character (0x80 and above), is treated as part of an identifier
     * for simplicity and speed, but this may be redefined. Any broken
     * utf-8 is not sanitized, thus 0x80 would be a valid identifier
     * token with utf-8 identifiers enabled, and otherwise it would be a
     * symbol token.
     *
     * The ID does a magic trick: It maps the lexeme to a very simple
     * and fast 32 bit hash code called a tag. The tag is emitted with
     * the id token and can be used for fast keyword lookup. The
     * hash tag is:
     *
     *     (length)(first char)(second char)(last char)
     *
     * where length is ASCII '0' .. '9' where any length overflow is an
     * arbitrary value, but such that the length is never longer than
     * the lexeme. The last char is the last char regardless of length.
     * For short identifiers, the second char may be the first char
     * duplicated, and the last char may be first char.
     *
     * This code is very simple to write by hand: "5whe" means while,
     * and can be used in a case switch before a strcmp with "while".
     * Conflicts are possible, but then several keywords are tested like
     * any other hash conflict. This keyword lookup is user driven, but
     * can follow example code quite straightforward.
     *
     * The lex_emit_id macro can be implemented to provide the above
     * lookup and inject a keyword token instead. By convention such
     * tokens have negative values to avoid conflicts with lexer
     * generated tokens.
     *
     * The ID also has a special role in prefixes and suffixes: C string
     * literals like (L"hello") and numeric literals like (42f) are
     * lexed as two tokens, one of which is an ID. The parser must
     * process this and observe absence of whitespace where such syntax
     * is relevant.
     *
     * While not specific to ID, the emitter macroes can be designed to
     * keep track of start of lines and end of whitespace and attach
     * state flags to each token (at line start, after whitespace). The
     * whitespace tokens can then be dropped. This might help parsing
     * things like suffixes efficiently.
     */
    LEX_TOK_ID,

    /*
     * C-int :: pos-dec-digit dec-digit *
     * Julia-int ::= dec-digit+
     *
     *    pos-dec-digit ::= '1'..'9'
     *    dec-digit ::= '0'..'9'
     *
     * Floating point numbers take precedence when possible so 00.10 is
     * always a deciaml floating point value when decimal floats are
     * enabled.
     *
     * The C-int is automatically enabled if C-octals are enabled, and
     * disabled otherwise. There is no specific Julia-int type - we just
     * use the terminology to represent integers with leading zeroes.
     *
     * Julia style integers accept leading zeroes. C style integers with
     * leading zeroes are consumed as C style octal numbers, so 0019 is
     * parsed as either 0019(Julia-int), or 001(C-octal), 9(C-int).
     *
     * Single digit '0' maps to octal when C-octals are enabled and to
     * Julia-int otherwise. (Yes, integers are not that simple, it
     * seems).
     *
     * Both C and Julia octal numbers (see octal token) can be active
     * simultaneously. This can be used to control leading zero
     * behavior, even if C-octal numbers are not part of the grammar
     * being parsed. For example, a language might use 0o777 octal
     * numbers and disallow 0777 integers. Enabling C-octals makes this
     * easy to detect (but should accept octal 0).
     *
     * There is no destinction between the styles in the int token, but
     * leading zeroes are easily detected in the lexeme.
     *
     * Constant suffixes like 1L are treated as 1(INT), and L(ID). The
     * same goes for other numeric values.
     *
     * Parser should check for leading zeroes and decide if it is valid,
     * a warning, or an error (it is in JSON). This also goes for float.
     *
     * Numericals, not limited to INT, may appear shorter than they are
     * due to buffer splits. Special recovery is required, but will only
     * happen just before EOS or EOB tokens (i.e. buffer split events).
     */
    LEX_TOK_INT,

    /*
     * float ::= (int ['.' dec-digits*] dec-exponent)
     *          | ([int] '.' dec-digits* [dec-exponent])
     *      dec-exponents ::= ('e' | 'E') ['+' | '-'] dec-digits*
     *      dec-digits ::= '0'..'9'
     *      int ::= dec-digits*
     *
     * Consumes a superset of C float representation without suffix.
     * Some invalid tokens such as 0.E are accepted. Valid tokens such
     * as 00.10 take precedence over octal numbers even if it is a
     * prefix, and the same is obviously true with respect to decimal
     * integers.
     *
     * JSON does not allow leading zeroes, and also not leading '.'.
     * This can easily be checked in the lexeme.
     *
     * The octal notation affecting integer leading zeroes is not
     * relevant to floats because floats take precedence over octal and
     * decimal int when containing '.', 'e' or 'E'.
     */
    LEX_TOK_FLOAT,

    /*
     * binary ::= (0b | 0B) ('0' | '1')*
     *
     * 0b100 or just 0b, parser must check that digits are present,
     * otherwise it may be interpreted as zero, just like octal zero
     * in C.
     *
     * Like 0X hex, 0B can be flagged out because Julia v0.3 does not
     * support uppercase 0B.
     */
    LEX_TOK_BINARY,

    /*
     * C-octal ::= 0 octal-digit*
     *     octal-digits ::= '0'..'7'
     *
     * Julia-octal ::= 0o octal-digits*
     *     octal-digits ::= '0'..'7'
     *
     * 0777 for C style octal numbers, or 0o777 for Julia syntax. Julia
     * v.0.3 does not allow uppercase 0O777, it would mean 0 * O777.
     *
     * When enabled, decimal floating points take precedence: 00.10 is
     * parsed as 00.10(decimal float), as per C standard.
     *
     * NOTE: It is possible for both styles to be active simultaneously.
     * This may be relevant in order to control handling of leading
     * zeroes in decimal integers.
     *
     * If C-octal numbers are flagged out, leading zeroes are mapped to
     * integers and the numerical value may change.  Julia behaves this
     * way. Nothing prevents support of both C and Julia octal numbers,
     * but leading zeroes will then be interpreted the C way - it is not
     * recommended to do this.
     */
    LEX_TOK_OCTAL,

    /*
     * hex ::= hex-int
     *     hex-digits ::= 'a'..'f'| 'A'..'f' | '0'..'9'
     *     hex-int ::= (0x | 0X) hex_digts*
     *
     * where hex_digits are customizable (e.g. all lower case), and hex
     * prefix 0x can be flagged to be lower case only (as in Julia).
     *
     * If hex floats are enabled, they take precedence:
     * 0x1.0(hex-float), if not, 0x1.0 will parse as: 0x1(hex) followed
     * by .0(decimal float).
     *
     * The lead prefix 0x may by flagged to be lower case only because
     * this is required by Julia v0.3 where 0X means 0 * X. Julia
     * accepts uppercase in the remaining hex digits (and exponent for
     * floats). This could possibly change in future versions.
     *
     * The zero length sequence (0x | 0X) is accepted and left to the
     * parser since the lexer emits a token for everything it sees.
     * Conceptually it may be interpreted as zero, equivalent to 0 being
     * both octal prefix and numeric 0 in C style octal representation.
     * Or it may be an error.
     */
    LEX_TOK_HEX,

    /*
     * hex_float ::= hex-int ['.' hex_digit*] hex-exponent
     *     hex-exponent ::= ('p' | 'P') ['+' | '-'] decimal-digit*
     *     decimal-digit ::= '0'..'9'
     *
     * A superset of IEEE-754-2008 Hexadecimal Floating Point notation.
     *
     * We require the exponent to be present, but does not ensure the
     * value is otherwise complete, e.g. 0x1p+ would be accepted. The p
     * is needed because otherwise 0x1.f could be accepted, and f is a
     * float suffix in C, and juxtapostion factor (0x1. * f) in Julia,
     * at least, that is one possible interpretation.
     *
     * The exponent can be flagged optional in which case 0x1.f will be
     * consumed as a single hex float toke as a single hex float token.
     * This may either simply be accepted in some grammars, or used to
     * provide an error message. If the exponent is required, 0x1.f will
     * be lexed as three tokens:
     *
     *     <'0x1'(hex int), '.'(op), 'f'(id)>.
     *
     * Thus it may be a good idea to allow the exponent to be optional
     * anyway and issue an error message or warning if the p is absent
     * later in the parsing stage.
     *
     * Note that, as per IEEE-754, the exponent is a decimal power of
     * two. In other words, the number of bits to shift the
     * (hexa)decimal point. Also note that it is p and not e because e
     * is a hex digit.
     */
    LEX_TOK_HEX_FLOAT,

    /*
     * blank ::= ('\t' | '\x20')+
     *
     * Longest run in buffer holding only '\t' and '\x20' (space).
     *
     * buffer splits may generate adjacent blanks depending on recovery
     * processing. (The same goes for other line oriented runs such as
     * string parts and comment parts).
     */
    LEX_TOK_BLANK,

    /* newline ::= '\r' | '\n' | '\r\n' | '\n\r'
     *
     * Will always appear, also inside strings and comments. Can be used
     * to track line starts and counts reliably as only one newline is
     * issued at a time, and it is issued everywhere, also in strings
     * and comments.
     *
     * May be preceeded by string escape token inside strings. This can
     * be interpreted as line continuation within strings specifically,
     * as is the case in Python and Javascript (and in C via
     * pre-processor).
     *
     * The LEX_TOK_STRING_NEWLINE is emitted inside strings so the ordinary
     * newline may be ignored in comments and other non-string content.
     */
    LEX_TOK_NEWLINE,
    LEX_TOK_STRING_NEWLINE,

    /*
     * string ::= string_start
     *            (string_part | string_escape |
     *                 string_ctrl | string_newline)*
     *            (string_end | string_unterminated)
     *
     * There are several optional string styles. They all start with
     * this token. The length and content provided details. Python
     * may start with """ or ''' and this token will then have length
     * 3 and three quotes as lexeme content. If the lexer exits before
     * string end token, the returned lexer mode will remember the
     * state and can be used for reentry - this also goes for comments.
     *
     * Strings can only contain part, escape, newline, and control
     * tokens, and either string unterminated or string end token
     * at last.
     */
    LEX_TOK_STRING_BEGIN,

    /* Longest run without control characters, without (\), without
     * newline, and without the relevant end delimiter. The run may be
     * shortened due to buffer splits. The part may, as an exception,
     * begin with an end delimiter character or a (\) if it was
     * preceeded by a string escape token. The escape character is
     * always (\). Strings that use "" or '' as escape will be treated
     * as start and end of separate strings. Strings that do not supoort
     * (\) should just treat escape as a part of the string.
     */
    LEX_TOK_STRING_PART,

    /*
     * This is always a single character token (\) and only happens
     * inside strings. See also string part token.
     */
    LEX_TOK_STRING_ESCAPE,

    /* This token is similar to string start. It may be absent at buffer
     * splits, but will then an unterminated string token will be used
     * just before the split event token.
     *
     * */
    LEX_TOK_STRING_END,

    /*
     * This is emitted before the buffer ends, or before unescaped
     * newlines for line oriented string types (the usual strings).
     * At buffer splits, recovery should clean it up. The returned
     * mode allow parsing to continue in a new buffer with a slight
     * content overlap.
     *
     * If string like ("hello, world!") in C, reaches end of line, it
     * may be continued" ("hello, \)newline(world!"). If this line
     * continuation is flagged out, this will lead to string
     * unterminated, even if not at end of buffer. For block strings
     * like """hello""", this only happens at end of buffer.
     */
    LEX_TOK_STRING_UNTERMINATED,

    /*
     *
     *     comment ::= comment_start
     *     (comment_part | ctrl | newline)*
     *     (comment_end | comment_unterminated)
     *
     *
     * Comments work like strings in most respects. They emit parts, and
     * control characters, but not escape characters, and cannot be
     * continued at end of line. Block comments are like python block
     * strings (''').
     *
     * Julia supports nested comments (#= ... #= =# =#). In this case
     * a new start token can be emitted before an end token. If the
     * parser exits due to buffer split, the mode has the nesting level
     * encoded so it can resumed in a new buffer.
     *
     * Line comments will have their end token just before newline, or
     * unterminated comment just before buffer split token (EOB or EOS).
     * (\) characters are consumed by the comment part tokens and do not
     * affect the end of any comment.
     *
     * Comment begin may include extra characters when a doc comment is
     * recognized. The emitter flags this. End comments are unaffected.
     */
    LEX_TOK_COMMENT_BEGIN,
    LEX_TOK_COMMENT_PART,
    LEX_TOK_COMMENT_END,
    LEX_TOK_COMMENT_UNTERMINATED,

    /*
     * Issued before ABORT token if nesting level is above a predefined
     * level. This is to protect against malicious and misguided
     * content, otherwise the nesting level counter could wrap and
     * generate a different interpretation, which could be bad. The
     * parser would probably do similar things with nested tokens.
     */
    LEX_TOK_COMMENT_DEEPLY_NESTED,


    /* Operators are all recognized single character symbols, or up to
     * four characters. The token value is the ASCII codes shifted 8
     * bits per extra character, by default, but the emitter macros
     * can redefine this. Values below 32 are reserved token types as
     * discussed above.
     *
     * What exactly represents an operator depends on what the lexer has
     * enabled.
     *
     * Printable ASCII symbols that are NOT recognized, are emitted as
     * the SYMBOL token and is always length 1. The value can be derived
     * from the lexeme, but not the token itself. This may be perfectly
     * fine for the parser, or it may be used to indicate an error.
     * There are no illegal characters per se.
     *
     * Non-printable ASCII characters that are not covered by newline or
     * blank, are emitted as CTRL tokens. These act the same as the
     * symbol token and may be used to indicate error, or to handle form
     * feed and other whitespace not handled by default. Unlike symbol,
     * however, CTRL also appear in strings and comments since they are
     * generally not allowed and this makes it easy to capture (there is
     * virtually no performance overhead in providing this service
     * unless attempting to parse a binary format).
     */

    /* Don't bleed into this range. */
    LEX_TOK_OPERATOR_BASE = 32,


    /*
     * Operators use ASCII range.
     * Compound operators use range 0x80 to 0x7fff
     * and possibly above for triple sequences.
     * Custom keywords are normally negative but can be mapped
     * to any other.
     *
     * The layout is designed for efficient table lookup.
     * Compound operators might benefit from remapping down to a smaller
     * range for compact lookup tables, but it depends on the parser.
     */
};

/*
 * Custom keyword token range is negative, and well below -99..0 where
 * special codes are reserved.
 */
#ifndef LEX_TOK_KW_BASE
#define LEX_TOK_KW_BASE -1000
#endif

#ifndef LEX_TOK_KW_NOT_FOUND
#define LEX_TOK_KW_NOT_FOUND LEX_TOK_ID
#endif


#ifdef LEX_DEBUG

#include <stdio.h>
#include <string.h>

static const char *lex_describe_token(long token)
{
    switch(token) {
    case LEX_TOK_BOM: return "BOM marker";
    case LEX_TOK_EOF: return "EOF";
    case LEX_TOK_EOS: return "buffer zero terminated";
    case LEX_TOK_EOB: return "buffer exhausted";
    case LEX_TOK_ABORT: return "abort";
    case LEX_TOK_CTRL: return "control";
    case LEX_TOK_STRING_CTRL: return "string control";
    case LEX_TOK_COMMENT_CTRL: return "comment control";
    case LEX_TOK_SYMBOL: return "symbol";
    case LEX_TOK_ID: return "identifier";
    case LEX_TOK_INT: return "integer";
    case LEX_TOK_FLOAT: return "float";
    case LEX_TOK_BINARY: return "binary";
    case LEX_TOK_OCTAL: return "octal";
    case LEX_TOK_HEX: return "hex";
    case LEX_TOK_HEX_FLOAT: return "hex float";
    case LEX_TOK_BLANK: return "blank";
    case LEX_TOK_NEWLINE: return "newline";
    case LEX_TOK_STRING_NEWLINE: return "string newline";
    case LEX_TOK_STRING_BEGIN: return "string begin";
    case LEX_TOK_STRING_PART: return "string part";
    case LEX_TOK_STRING_END: return "string end";
    case LEX_TOK_STRING_ESCAPE: return "string escape";
    case LEX_TOK_STRING_UNTERMINATED: return "unterminated string";
    case LEX_TOK_COMMENT_BEGIN: return "comment begin";
    case LEX_TOK_COMMENT_PART: return "comment part";
    case LEX_TOK_COMMENT_END: return "comment end";
    case LEX_TOK_COMMENT_UNTERMINATED: return "unterminated comment";
    case LEX_TOK_COMMENT_DEEPLY_NESTED: return "deeply nested comment";

    default:
        if (token < LEX_TOK_EOF) {
            return "keyword";
        }
        if (token < 32) {
            return "undefined";
        }
        if (token < 0x100L) {
            return "operator";
        }
        if (token < 0x10000L) {
            return "compound operator";
        }
        if (token < 0x1000000L) {
            return "tricompound operator";
        }
        if (token < 0x7f0000000L) {
            return "quadcompound operator";
        }
        return "reserved";
    }
}

static void lex_fprint_token(FILE *fp,
        long token,
        const char *first, const char *last,
        int line, int pos)
{
    char buf[10];
    const char *lexeme = first;
    int len = (int)(last - first);
    switch (token) {
    case LEX_TOK_EOS:
    case LEX_TOK_CTRL:
        sprintf(buf, "^%02x", (int)*first);
        lexeme = buf;
        len = strlen(buf);
        break;
    default:
        break;
    }
    fprintf(fp, "%04d:%03d %s (0x%lx): `%.*s`\n",
            line, pos, lex_describe_token(token), token, len, lexeme);
}

#define lex_print_token(token, first, last, line, pos)                  \
        lex_fprint_token(stdout, token, first, last, line, pos)

#else /* LEX_DEBUG */

#define lex_describe_token(token) "debug not available"
#define lex_fprint_token(fp, token, first, last, line, pos) ((void)0)
#define lex_print_token(token, first, last, line, pos) ((void)0)

#endif /* LEX_DEBUG */


#endif /* LEX_TOKENS_H */

