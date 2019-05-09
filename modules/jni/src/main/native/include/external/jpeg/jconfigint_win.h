/* libjpeg-turbo build number */
#define BUILD  "20180831"

/* Compiler's inline keyword */


/* How to obtain function inlining. */
#if defined(__GNUC__)
#define INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define INLINE __forceinline
#else
#define INLINE
#endif


/* Define to the full name of this package. */
#define PACKAGE_NAME  "libjpeg-turbo"

/* Version number of package */
#define VERSION  "2.0.0"

/* The size of `size_t', as computed by sizeof. */
#if (__WORDSIZE==64)
#define SIZEOF_SIZE_T 8
#else
#define SIZEOF_SIZE_T 4
#endif


/* Define if your compiler has __builtin_ctzl() and sizeof(unsigned long) == sizeof(size_t). */


/* Define to 1 if you have the <intrin.h> header file. */


#if defined(_MSC_VER) && defined(HAVE_INTRIN_H)
#if (SIZEOF_SIZE_T == 8)
#define HAVE_BITSCANFORWARD64
#elif (SIZEOF_SIZE_T == 4)
#define HAVE_BITSCANFORWARD
#endif
#endif
