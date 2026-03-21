 /* Note: only one hash table can be implemented a single file. */


/*
 * The generic hash table is designed to make the key length optional
 * and we do not need it because our key is a terminated token list.
 *
 * The token list avoids having to allocated a new string and the
 * associated issues of memory management. In most cases the search key
 * is also a similar token list.
 *
 * However, on occasion we need to look up an unparsed string of dot
 * separated scopes (nested_flatbuffer attributes). This is not
 * trivially possible without reverting to allocating the strings.
 * We could parse the attribute into tokens but it is also non-trivial
 * because the token buffer breaks pointers when reallocating and
 * the parse output is considered read-only at this point.
 *
 * We can however, use a trick to overcome this because the hash table
 * does not enforce that the search key has same representation as the
 * stored key. We can use the key length to switch between key types.
 *
 * When the key is paresed to a token list:
 *
 *   enemy: MyGame . Example.Monster
 *
 * the spaces in dots may be ignored by the parser.
 * Spaces must be handled explicitly or disallowed when the key is
 * parsed as an attribute string (only the quoted content):
 *
 *   (nested_flatbuffer:"MyGame.Example.Monster")
 *
 * vs
 *
 *   (nested_flatbuffer:"MyGame . Example.Monster")
 *
 * Googles flatc allows spaces in the token stream where dots are
 * operators, but not in attribute strings which are supposed to
 * be unique so we follow that convention.
 *
 * On both key representations, preprocessing must strip the trailing
 * symbol stored within the scope before lookup - minding that this
 * lookup only finds the scope itself. For token lists this can be
 * done by either zero terminating the list early, or by issuing
 * a negative length (after cast to int) of elements to consider. For
 * string keys the key length should be made to the length to be
 * considered.
 *
 * If the scope string is zero length, a null key should be issued
 * with zero length. This is indistinguishly from a null length token
 * list - both indicating a global scope - null thus being a valid key.
 * 
 * Note: it is important to not use a non-null zero length string
 * as key.
 */

#include "../symbols.h"

static inline size_t scope_hash(const void *s, size_t len);
#define HT_HASH_FUNCTION scope_hash

#include "hash/hash_table_def.h"
DEFINE_HASH_TABLE(fb_scope_table)
#include "hash/hash_table_impl.h"

/* Null is a valid key used for root scopes. */
static inline int ht_match(const void *key, size_t len, fb_scope_t *scope)
{
    const fb_ref_t *name = scope->name;
    int count = (int)len;
    size_t n1, n2, i;

    /* Note: `name` may be null here - this is the global scope name. */
    if (count <= 0) {
        const fb_ref_t *keyname = key;
        /*
         * If count is negative, this is the token count of the key
         * which may have suffix to be ignored, otherwise the key is the
         * full list.
         */
        /* `key` is a ref list (a list of tokens). */
        while (name && keyname) {
            n1 = (size_t)name->ident->len;
            n2 = (size_t)keyname->ident->len;
            if (n1 != n2 || strncmp(name->ident->text, keyname->ident->text, n1)) {
                return 0;
            }
            name = name->link;
            keyname = keyname->link;
            if (++count == 0) {
                return name == 0;
            }
        }
        if (name || keyname) {
            return 0;
        }
        return 1;
    } else {
        /* `key` is a dotted string. */
        const char *s1, *s2 = key;
        while (name) {
            s1 = name->ident->text;
            n1 = (size_t)name->ident->len;
            if (n1 > len) {
                return 0;
            }
            for (i = 0; i < n1; ++i) {
                if (s1[i] != s2[i]) {
                    return 0;
                }
            }
            if (n1 == len) {
                return name->link == 0;
            }
            if (s2[i] != '.') {
                return 0;
            }
            len -= n1 + 1;
            s2 += n1 + 1;
            name = name->link;
        }
        return 0;
    }
}

static inline const void *ht_key(fb_scope_t *scope)
{
    return scope->name;
}

static inline size_t ht_key_len(fb_scope_t *scope)
{
    (void)scope;
    /*
     * Must be zero because the result is passed to ht_match
     * when comparing two stored items for hash conflicts.
     * Only external lookup keys can be non-zero.
     */
    return 0;
}

static inline size_t scope_hash(const void *key, size_t len)
{
    size_t h = 0, i;
    int count = (int)len;

    if (count <= 0) {
        const fb_ref_t *name = key;

        while (name) {
            h ^= ht_strn_hash_function(name->ident->text, (size_t)name->ident->len);
            h = ht_int_hash_function((void *)h, 0);
            name = name->link;
            if (++count == 0) {
                break;
            }
        }
        return h;
    } else {
        const char *s = key;
        for (;;) {
            for (i = 0; i < len; ++i) {
                if (s[i] == '.') {
                    break;
                }
            }
            h ^= ht_strn_hash_function(s, i);
            h = ht_int_hash_function((void *)h, 0);
            if (i == len) {
                break;
            }
            len -= i + 1;
            s += i + 1;
        }
        return h;
    }
}
