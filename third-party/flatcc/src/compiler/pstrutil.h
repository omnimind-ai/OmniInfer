#ifndef PSTRUTIL_H
#define PSTRUTIL_H

#include <ctype.h> /* toupper */


/*
 * NOTE: unlike strncpy, we return the first character, and we do not
 * pad up to n. Same applies to related functions.
 */

/* `strnlen` not widely supported. */
static inline size_t pstrnlen(const char *s, size_t max_len)
{
    const char *end = memchr (s, 0, max_len);
    return end ? (size_t)(end - s) : max_len;
}

static inline char *pstrcpyupper(char *dst, const char *src) {
    char *p = dst;
    while (*src) {
        *p++ = (char)toupper(*src++);
    }
    *p = '\0';
    return dst;
}

static inline char *pstrncpyupper(char *dst, const char *src, size_t n) {
    size_t i;
    for (i = 0; i < n && src[i]; ++i) {
        dst[i] = (char)toupper(src[i]);
    }
    if (i < n) {
        dst[i] = '\0';
    }
    return dst;
}

static inline char *pstrtoupper(char *dst) {
    char *p;
    for (p = dst; *p; ++p) {
        *p = (char)toupper(*p);
    }
    return dst;
}

static inline char *pstrntoupper(char *dst, size_t n) {
    size_t i;
    for (i = 0; i < n && dst[i]; ++i) {
        dst[i] = (char)toupper(dst[i]);
    }
    return dst;
}

#undef strnlen
#define strnlen pstrnlen

#endif /*  PSTRUTIL_H */
