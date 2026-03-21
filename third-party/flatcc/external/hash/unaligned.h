#ifndef UNALIGNED_H
#define UNALIGNED_H

/*
 * This is a simplified version of portable/punaligned.h that does not depend on
 * endian detection, but which assumes x86 is always little endian.
 * Include the portable version for better precision.
 */

#ifndef unaligned_read_le16toh

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)

#define unaligned_read_le16toh(p) (*(uint16_t*)(p))
#define unaligned_read_le32toh(p) (*(uint32_t*)(p))
#define unaligned_read_le64toh(p) (*(uint64_t*)(p))

#else

#define unaligned_read_le16toh(p)  (                                        \
        (((uint16_t)(((uint8_t *)(p))[0])) <<  0) |                         \
        (((uint16_t)(((uint8_t *)(p))[1])) <<  8))

#define unaligned_read_le32toh(p)  (                                        \
        (((uint32_t)(((uint8_t *)(p))[0])) <<  0) |                         \
        (((uint32_t)(((uint8_t *)(p))[1])) <<  8) |                         \
        (((uint32_t)(((uint8_t *)(p))[2])) << 16) |                         \
        (((uint32_t)(((uint8_t *)(p))[3])) << 24))

#define unaligned_read_le64toh(p)  (                                        \
        (((uint64_t)(((uint8_t *)(p))[0])) <<  0) |                         \
        (((uint64_t)(((uint8_t *)(p))[1])) <<  8) |                         \
        (((uint64_t)(((uint8_t *)(p))[2])) << 16) |                         \
        (((uint64_t)(((uint8_t *)(p))[3])) << 24) |                         \
        (((uint64_t)(((uint8_t *)(p))[4])) << 32) |                         \
        (((uint64_t)(((uint8_t *)(p))[5])) << 40) |                         \
        (((uint64_t)(((uint8_t *)(p))[6])) << 48) |                         \
        (((uint64_t)(((uint8_t *)(p))[7])) << 56))
#endif
#endif

#endif /* UNALIGNED_H */
