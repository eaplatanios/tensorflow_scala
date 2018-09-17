/* libjpeg-turbo build number */
#define BUILD "20161115"

/* Compiler's inline keyword */


/* How to obtain function inlining. */
#define INLINE inline __attribute__((always_inline))

/* Define to the full name of this package. */
#define PACKAGE_NAME "libjpeg-turbo"

/* Version number of package */
#define VERSION "1.5.1"

/* The size of `size_t', as computed by sizeof. */
#if (__WORDSIZE==64 && !defined(__native_client__))
#define SIZEOF_SIZE_T 8
#else
#define SIZEOF_SIZE_T 4
#endif

