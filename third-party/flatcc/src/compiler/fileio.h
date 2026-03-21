#ifndef FILES_H
#define FILES_H

#include <stdlib.h>

/*
 * Returns an allocated copy of the path truncated to len if len is
 * shorter. Free returned string subsequently. Also truncates to less
 * than len if path contains null characters.
 */
char *__flatcc_fb_copy_path_n(const char *path, size_t len);
#define fb_copy_path_n __flatcc_fb_copy_path_n

/* Returns an allocated copy of path. Free returned string subsequently. */
char *__flatcc_fb_copy_path(const char *path);
#define fb_copy_path __flatcc_fb_copy_path

/*
 * Joins two paths. The prefix can optionally be null.
 * Free returned string subsequently. If `path_sep` is true, prefix is
 * separated from suffix with a path separator if not already present.
 */
char *__flatcc_fb_create_join_path_n(const char *prefix, size_t prefix_len,
        const char *suffix, size_t suffix_len, const char *ext, int path_sep);
#define fb_create_join_path_n __flatcc_fb_create_join_path_n

char *__flatcc_fb_create_join_path(const char *prefix, const char * suffix, const char *ext, int path_sep);
#define fb_create_join_path __flatcc_fb_create_join_path

/* Adds extension to path in a new copy. */
char *__flatcc_fb_create_path_ext_n(const char *path, size_t path_len, const char *ext);
#define fb_create_path_ext_n __flatcc_fb_create_path_ext_n

char *__flatcc_fb_create_path_ext(const char *path, const char *ext);
#define fb_create_path_ext __flatcc_fb_create_path_ext

/*
 * Creates a path with spaces escaped in a sort of gcc/Gnu Make
 * compatible way, primarily for use with dependency files.
 *
 * http://clang.llvm.org/doxygen/DependencyFile_8cpp_source.html
 *
 * We should escape a backslash only if followed by space.
 * We should escape a space in all cases.
 * We ought to handle to #, but don't because gcc fails to do so.
 *
 * This is dictated by how clang and gcc generates makefile
 * dependency rules for gnu make.
 *
 * This is not intended for strings used for system calls, but rather
 * for writing to files where a quoted format is not supported.
 *
 */
char *__flatcc_fb_create_make_path_n(const char *path, size_t path_len);
#define fb_create_make_path_n __flatcc_fb_create_make_path_n

char *__flatcc_fb_create_make_path(const char *path);
#define fb_create_make_path __flatcc_fb_create_make_path

/*
 * Creates a new filename stripped from path prefix and optional ext
 * suffix. Free returned string subsequently.
 */
char *__flatcc_fb_create_basename(const char *path, size_t len, const char *ext);
#define fb_create_basename __flatcc_fb_create_basename

/* Free returned buffer subsequently. Stores file size in `size_out` arg.
 * if `max_size` is 0 the file as read regardless of size, otherwise
 * if the file size exceeds `max_size` then `size_out` is set to the
 * actual size and null is returend. */
char *__flatcc_fb_read_file(const char *filename, size_t max_size, size_t *size_out);
#define fb_read_file __flatcc_fb_read_file


/*
 * Returns offset into source path representing the longest suffix
 * string with no path separator.
 */
size_t __flatcc_fb_find_basename(const char *path, size_t len);
#define fb_find_basename __flatcc_fb_find_basename

/* Returns input length or length reduced by ext len if ext is a proper suffix. */
size_t __flatcc_fb_chomp(const char *path, size_t len, const char *ext);
#define fb_chomp __flatcc_fb_chomp

#endif /* FILES_H */
