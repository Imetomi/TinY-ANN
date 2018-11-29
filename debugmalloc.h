#ifndef DEBUGMALLOC_H
#define DEBUGMALLOC_H

/* 
 * Ezek azok a makrok, amelyek biztositjak, hogy a sajat
 * fuggvenyek hivodjanak, amikor malloc() vagy free()
 * szerepel a programban.
 * 
 * Semmi egyeb teendo nincsen, csak az, hogy ezt a header
 * fajlt be kell szerkeszteni _minden_ forraskodba. A szokasos
 * formaban megadott malloc() es free() hivasok helyett ez a
 * valtozat fog automatikusan mindig meghivodni a makrok miatt.
 * 
 * Fontos, hogy ezt az include fajlt a project OSSZES
 * forras fajljaban hasznalni kell! Siman, a beepitett
 * free()-vel nem lehet felszabaditani egy olyan teruletet,
 * amelyet ez a malloc() foglalt! Mas a pointer.
 * 
 * Ha szukseg van arra, hogy a malloc() es a free() cime
 * kepezheto legyen, akkor a HASZNALOM_A_MALLOC_FREE_POINTERET
 * makrot definialni kell az #include "debugmalloc.h" sor
 * elott. Ilyenkor egy butabb valtozatot fog hasznalni,
 * amely viszont kompatibilis prototipussal rendelkezik.
 */

#include <stdlib.h>
#include <stdbool.h>

#ifdef HASZNALOM_A_MALLOC_FREE_POINTERET
    #define malloc debugmalloc_malloc
    #define calloc debugmalloc_calloc
    #define realloc debugmalloc_realloc
    #define free debugmalloc_free
#else
    #define malloc(X) debugmalloc_malloc_full(X, "malloc", #X, __FILE__, __LINE__, false)
    #define calloc(X,Y) debugmalloc_malloc_full(X*Y, "calloc", #X ", " #Y, __FILE__, __LINE__, true)
    #define realloc(P,X) debugmalloc_realloc_full(P, X, "realloc", #X, __FILE__, __LINE__)
    #define free(P) debugmalloc_free_full(P, "free", __FILE__, __LINE__)
#endif

/*
 * Ezzel a fuggvennyel lehet megadni azt, hogy az stderr helyett
 * egy fajlba irja az uzeneteit a debugmalloc. Ures fajlnev ("")
 * eseten visszaallitja stderr-re.
 */
void debugmalloc_naplofajl(char const *nev);

/*
 * Ezt meghivva egy listat keszit az aktualisan lefoglalt
 * memoriateruletekrol.
 */
void debugmalloc_dump();

/* 
 * Ezek a fuggvenyek vannak a szokasos malloc() es free()
 * helyett. Nem kell oket hasznalni kozvetlenul; a makrok
 * altal a rendes malloc() es free() ezekre cserelodik ki.
 */
void *debugmalloc_malloc_full(size_t meret, char const *fv, char const *cel, char const *file, unsigned line, bool zero);
void *debugmalloc_realloc_full(void *regimem, size_t newsize, char const *fv, char const *cel, char const *file, unsigned line);
void debugmalloc_free_full(void *mem, char const *fv, char const *file, unsigned line);

void *debugmalloc_malloc(size_t meret);
void *debugmalloc_calloc(size_t nmemb, size_t meret);
void *debugmalloc_realloc(void *regimem, size_t meret);
void debugmalloc_free(void *mem);

#endif  /* DEBUGMALLOC_H */
