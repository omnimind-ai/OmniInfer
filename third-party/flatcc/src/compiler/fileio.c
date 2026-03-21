#include <string.h>
#include <stdio.h>

/* Ensures portable headers are included such as inline. */
#include "config.h"
#include "fileio.h"
#include "pstrutil.h"

char *fb_copy_path_n(const char *path, size_t len)
{
    size_t n;
    char *s;

    n = strnlen(path, len);
    if ((s = malloc(n + 1))) {
        memcpy(s, path, n);
        s[n] = '\0';
    }
    return s;
}

char *fb_copy_path(const char *path)
{
    size_t n;
    char *s;

    n = strlen(path);
    if ((s = malloc(n + 1))) {
        memcpy(s, path, n);
        s[n] = '\0';
    }
    return s;
}

size_t fb_chomp(const char *path, size_t len, const char *ext)
{
    size_t ext_len = ext ? strlen(ext) : 0;
    if (len > ext_len && 0 == strncmp(path + len - ext_len, ext, ext_len)) {
        len -= ext_len;
    }
    return len;
}

char *fb_create_join_path_n(const char *prefix, size_t prefix_len,
        const char *suffix, size_t suffix_len, const char *ext, int path_sep)
{
    char *path;
    size_t ext_len = ext ? strlen(ext) : 0;
    size_t n;

    if (!prefix ||
            (suffix_len > 0 && (suffix[0] == '/' || suffix[0] == '\\')) ||
            (suffix_len > 1 && suffix[1] == ':')) {
        prefix_len = 0;
    }
    if (path_sep && (prefix_len == 0 ||
            (prefix[prefix_len - 1] == '/' || prefix[prefix_len - 1] == '\\'))) {
        path_sep = 0;
    }
    path = malloc(prefix_len + !!path_sep + suffix_len + ext_len + 1);
    if (!path) {
        return 0;
    }
    n = 0;
    if (prefix_len > 0) {
        memcpy(path, prefix, prefix_len);
        n += prefix_len;
    }
    if (path_sep) {
        path[n++] = '/';
    }
    memcpy(path + n, suffix, suffix_len);
    n += suffix_len;
    memcpy(path + n, ext, ext_len);
    n += ext_len;
    path[n] = '\0';
    return path;
}

char *fb_create_join_path(const char *prefix, const char *suffix, const char *ext, int path_sep)
{
    return fb_create_join_path_n(prefix, prefix ? strlen(prefix) : 0,
            suffix, suffix ? strlen(suffix) : 0, ext, path_sep);
}

char *fb_create_path_ext_n(const char *path, size_t path_len, const char *ext)
{
    return fb_create_join_path_n(0, 0, path, path_len, ext, 0);
}

char *fb_create_path_ext(const char *path, const char *ext)
{
    return fb_create_join_path(0, path, ext, 0);
}

char *fb_create_make_path_n(const char *path, size_t len)
{
    size_t i, j, n;
    char *s;

    if (len == 1 && (path[0] == ' ' || path[0] == '\\')) {
        if (!(s = malloc(3))) {
            return 0;
        }
        s[0] = '\\';
        s[1] = path[0];
        s[2] = '\0';
        return s;
    }
    if (len <= 1) {
        return fb_copy_path_n(path, len);
    }
    for (i = 0, n = len; i < len - 1; ++i) {
        if (path[i] == '\\' && path[i + 1] == ' ') {
            ++n;
        }
        n += path[i] == ' ';
    }
    n += path[i] == ' ';
    if (!(s = malloc(n + 1))) {
        return 0;
    }
    for (i = 0, j = 0; i < len - 1; ++i, ++j) {
        if (path[i] == '\\' && path[i + 1] == ' ') {
            s[j++] = '\\';
        }
        if (path[i] == ' ') {
            s[j++] = '\\';
        }
        s[j] = path[i];
    }
    if (path[i] == ' ') {
        s[j++] = '\\';
    }
    s[j++] = path[i];
    s[j] = 0;
    return s;
}

char *fb_create_make_path(const char *path)
{
    return fb_create_make_path_n(path, strlen(path));
}

size_t fb_find_basename(const char *path, size_t len)
{
    char *p = (char *)path;

    p += len;
    while(p != path) {
        --p;
        if (*p == '/' || *p == '\\') {
            ++p;
            break;
        }
    }
    return (size_t)(p - path);
}

char *fb_create_basename(const char *path, size_t len, const char *ext)
{
    size_t pos;
    char *s;

    pos = fb_find_basename(path, len);
    path += pos;
    len -= pos;
    len = fb_chomp(path, len, ext);
    if ((s = malloc(len + 1))) {
        memcpy(s, path, len);
        s[len] = '\0';
    }
    return s;
}

char *fb_read_file(const char *filename, size_t max_size, size_t *size_out)
{
    FILE *fp;
    long k;
    size_t size, pos, n, _out;
    char *buf;

    size_out = size_out ? size_out : &_out;

    fp = fopen(filename, "rb");
    size = 0;
    buf = 0;

    if (!fp) {
        goto fail;
    }
    fseek(fp, 0L, SEEK_END);
    k = ftell(fp);
    if (k < 0) goto fail;
    size = (size_t)k;
    *size_out = size;
    if (max_size > 0 && size > max_size) {
        goto fail;
    }
    rewind(fp);
    buf = malloc(size ? size : 1);
    if (!buf) {
        goto fail;
    }
    pos = 0;
    while ((n = fread(buf + pos, 1, size - pos, fp))) {
        pos += n;
    }
    if (pos != size) {
        goto fail;
    }
    fclose(fp);
    *size_out = size;
    return buf;

fail:
    if (fp) {
        fclose(fp);
    }
    if (buf) {
        free(buf);
    }
    *size_out = size;
    return 0;
}
