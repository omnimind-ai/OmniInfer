#ifndef CATALOG_H
#define CATALOG_H

#include <stdlib.h>
#include "symbols.h"

/* Helper to build more intuitive schema data with fully qualified names. */


typedef struct entry entry_t;
typedef entry_t object_entry_t;
typedef entry_t enum_entry_t;
typedef entry_t service_entry_t;
typedef struct scope_entry scope_entry_t;

struct entry {
    fb_compound_type_t *ct;
    char *name;
};

struct scope_entry {
    fb_scope_t *scope;
    char *name;
};

typedef struct catalog catalog_t;

struct catalog {
    int qualify_names;
    int nobjects;
    int nenums;
    int nservices;
    size_t name_table_size;
    object_entry_t *objects;
    enum_entry_t *enums;
    service_entry_t *services;
    char *name_table;
    object_entry_t *next_object;
    enum_entry_t *next_enum;
    service_entry_t *next_service;
    char *next_name;
    fb_schema_t *schema;
};

#include <stdio.h>

static void count_symbol(void *context, fb_symbol_t *sym)
{
    catalog_t *catalog = context;
    fb_ref_t *scope_name;
    size_t n = 0;
    fb_compound_type_t *ct;
    
    if (!(ct = get_compound_if_visible(catalog->schema, sym))) {
        return;
    }

    /*
     * Find out how much space the name requires. We must store each
     * name in full for sorting because comparing a variable number of
     * parent scope names is otherwise tricky.
     */
    if (catalog->qualify_names) {
        scope_name = ct->scope->name;
        while (scope_name) {
            /* + 1 for '.'. */
            n += (size_t)scope_name->ident->len + 1;
            scope_name = scope_name->link;
        }
    }
    /* + 1 for '\0'. */
    n += (size_t)sym->ident->len + 1;
    catalog->name_table_size += n;

    switch (sym->kind) {
    case fb_is_struct:
    case fb_is_table:
        ++catalog->nobjects;
        break;
    case fb_is_union:
    case fb_is_enum:
        ++catalog->nenums;
        break;
    case fb_is_rpc_service:
        ++catalog->nservices;
        break;
    default: return;
    }
}

static void install_symbol(void *context, fb_symbol_t *sym)
{
    catalog_t *catalog = context;
    fb_ref_t *scope_name;
    int n = 0;
    char *s, *name;
    fb_compound_type_t *ct;
    
    if (!(ct = get_compound_if_visible(catalog->schema, sym))) {
        return;
    }

    s = catalog->next_name;
    name = s;
    if (catalog->qualify_names) {
        scope_name = ct->scope->name;
        while (scope_name) {
            n = (int)scope_name->ident->len;
            memcpy(s, scope_name->ident->text, (size_t)n);
            s += n;
            *s++ = '.';
            scope_name = scope_name->link;
        }
    }
    n = (int)sym->ident->len;
    memcpy(s, sym->ident->text, (size_t)n);
    s += n;
    *s++ = '\0';
    catalog->next_name = s;

    switch (sym->kind) {
    case fb_is_struct:
    case fb_is_table:
        catalog->next_object->ct = (fb_compound_type_t *)sym;
        catalog->next_object->name = name;
        catalog->next_object++;
        break;
    case fb_is_union:
    case fb_is_enum:
        catalog->next_enum->ct = (fb_compound_type_t *)sym;
        catalog->next_enum->name = name;
        catalog->next_enum++;
        break;
    case fb_is_rpc_service:
        catalog->next_service->ct = (fb_compound_type_t *)sym;
        catalog->next_service->name = name;
        catalog->next_service++;
        break;
    default: break;
    }
}

static void count_symbols(void *context, fb_scope_t *scope)
{
    fb_symbol_table_visit(&scope->symbol_index, count_symbol, context);
}

static void install_symbols(void *context, fb_scope_t *scope)
{
    fb_symbol_table_visit(&scope->symbol_index, install_symbol, context);
}

static int compare_entries(const void *x, const void *y)
{
    return strcmp(((const entry_t *)x)->name, ((const entry_t *)y)->name);
}

static void sort_entries(entry_t *entries, int count)
{
    int i;

    qsort(entries, (size_t)count, sizeof(entries[0]), compare_entries);

    for (i = 0; i < count; ++i) {
        entries[i].ct->export_index = (size_t)i;
    }
}

static void clear_catalog(catalog_t *catalog)
{
    if (catalog->objects) {
        free(catalog->objects);
    }
    if (catalog->enums) {
        free(catalog->enums);
    }
    if (catalog->services) {
        free(catalog->services);
    }
    if (catalog->name_table) {
        free(catalog->name_table);
    }
    memset(catalog, 0, sizeof(*catalog));
}

static int build_catalog(catalog_t *catalog, fb_schema_t *schema, int qualify_names, fb_scope_table_t *index)
{
    memset(catalog, 0, sizeof(*catalog));
    catalog->qualify_names = qualify_names;
    catalog->schema = schema;

    /* Build support datastructures before export. */
    fb_scope_table_visit(index, count_symbols, catalog);
    catalog->objects = calloc((size_t)catalog->nobjects, sizeof(catalog->objects[0]));
    catalog->enums = calloc((size_t)catalog->nenums, sizeof(catalog->enums[0]));
    catalog->services = calloc((size_t)catalog->nservices, sizeof(catalog->services[0]));
    catalog->name_table = malloc(catalog->name_table_size);
    catalog->next_object = catalog->objects;
    catalog->next_enum = catalog->enums;
    catalog->next_service = catalog->services;
    catalog->next_name = catalog->name_table;
    if ((!catalog->objects && catalog->nobjects > 0) ||
        (!catalog->enums && catalog->nenums > 0) ||
        (!catalog->services && catalog->nservices > 0) ||
        (!catalog->name_table && catalog->name_table_size > 0)) {
        clear_catalog(catalog);
        return -1;
    }
    fb_scope_table_visit(index, install_symbols, catalog);
    /* Presort objects and enums because the sorted index is required in Type tables. */
    sort_entries(catalog->objects, catalog->nobjects);
    sort_entries(catalog->enums, catalog->nenums);
    sort_entries(catalog->services, catalog->nservices);
    return 0;
}

#endif /* CATALOG_H */
