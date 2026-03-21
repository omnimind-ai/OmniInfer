#include "coerce.h"

/*
 * Be aware that some value variants represents actual values (e.g.
 * vt_int), and others represent a type (e.g. vt_scalar) which holds a
 * type identifier token. Here we implicitly expect a vt_scalar type as
 * first argument, but only receive the token. The second argument is a
 * value literal.  Our job is to decide if the value fits within the
 * given type. Our internal representation already ensures that value
 * fits within a 64bit signed or unsigned integer, or double; otherwise
 * the parser would have set vt_invalid type on the value.
 *
 * If the value is invalid, success is returned because the
 * error is presumably already generated. If the value is some other
 * type than expect, an error is generated.
 *
 * Symbolic names are not allowed as values here.
 *
 * Converts positive integers to signed type and unsigned integers to
 * signed type, integers to floats and floats to integers.
 *
 * Optionally allows 1 to be assigned as true and 0 as false, and vice
 * versa when allow_boolean_conversion is enabled.
 *
 * Returns 0 on success, -1 on error.
 */
int fb_coerce_scalar_type(fb_parser_t *P, fb_symbol_t *sym, fb_scalar_type_t st, fb_value_t *value)
{
    double d;
    float f;

    if (!value->type) {
        return 0;
    }
    /*
     * The parser only produces negative vt_int values, which simplifies
     * the logic, but to make this operation robust against multiple
     * coercion steps, we first convert back to uint if the assumption turns
     * out false.
     */
    if (value->type == vt_int && value->i >= 0) {
        value->type = vt_uint;
        value->u = (uint64_t)value->i;
    }
    if (value->type == vt_invalid) {
        /* Silently ignore past errors. */
        return 0;
    }
    if (value->type == vt_bool && st != fb_bool  && P->opts.allow_boolean_conversion) {
        value->type = vt_uint;
        value->u = (uint64_t)value->b;
        assert(value->u == 1 || value->u == 0);
    }
    switch (st) {
    case fb_ulong:
        if (value->type != vt_uint) {
            error_sym(P, sym, "64-bit uint32_t type only accepts unsigned integers");
            value->type = vt_invalid;
            return -1;
        }
        return 0;
    case fb_uint:
        if (value->type != vt_uint) {
            error_sym(P, sym, "32-bit unsigned int type only accepts unsigned integers");
            value->type = vt_invalid;
            return -1;
        }
        if (value->u > UINT32_MAX) {
            error_sym(P, sym, "32-bit unsigned int overflow");
            value->type = vt_invalid;
            return -1;
        }
        return 0;
    case fb_ushort:
        if (value->type != vt_uint) {
            error_sym(P, sym, "16-bit unsigned short type only accepts unsigned integers");
            value->type = vt_invalid;
            return -1;
        }
        if (value->u > UINT16_MAX) {
            error_sym(P, sym, "16-bit unsigned short overflow");
            value->type = vt_invalid;
            return -1;
        }
        return 0;
    case fb_char:
        /* Although C treats char as signed by default, flatcc treats it as unsigned. */
    case fb_ubyte:
        if (value->type != vt_uint) {
            error_sym(P, sym, "8-bit unsigned byte type only accepts unsigned integers");
            value->type = vt_invalid;
            return -1;
        }
        if (value->u > UINT8_MAX) {
            error_sym(P, sym, "8-bit unsigned byte overflow");
            value->type = vt_invalid;
            return -1;
        }
        return 0;
    case fb_long:
        if (value->type == vt_int) {
            /* Native format is always ok, or parser would have failed. */
            return 0;
        }
        if (value->type == vt_uint) {
            if (value->u >= (1ULL << 63)) {
                error_sym(P, sym, "64-bit signed int32_t overflow");
                value->type = vt_invalid;
                return -1;
            }
            value->i = (int64_t)value->u;
            value->type = vt_int;
            return 0;
        }
        error_sym(P, sym, "64-bit int32_t type only accepts integers");
        value->type = vt_invalid;
        return -1;
    case fb_int:
        if (value->type == vt_int) {
            if (value->i < INT32_MIN) {
                error_sym(P, sym, "32-bit signed int underflow");
                value->type = vt_invalid;
                return -1;
            }
            return 0;
        }
        if (value->type == vt_uint) {
            if (value->i > INT32_MAX) {
                error_sym(P, sym, "32-bit signed int overflow");
                value->type = vt_invalid;
                return -1;
            }
            value->i = (int64_t)value->u;
            value->type = vt_int;
            return 0;
        }
        error_sym(P, sym, "32-bit signed int type only accepts integers");
        value->type = vt_invalid;
        return -1;
    case fb_short:
        if (value->type == vt_int) {
            if (value->i < INT16_MIN) {
                error_sym(P, sym, "16-bit signed short underflow");
                value->type = vt_invalid;
                return -1;
            }
            return 0;
        }
        if (value->type == vt_uint) {
            if (value->i > INT16_MAX) {
                error_sym(P, sym, "16-bit signed short overflow");
                value->type = vt_invalid;
                return -1;
            }
            value->i = (int64_t)value->u;
            value->type = vt_int;
            return 0;
        }
        error_sym(P, sym, "16-bit signed short type only accepts integers");
        value->type = vt_invalid;
        return -1;
    case fb_byte:
        if (value->type == vt_int) {
            if (value->i < INT8_MIN) {
                error_sym(P, sym, "8-bit signed byte underflow");
                value->type = vt_invalid;
                return -1;
            }
            return 0;
        }
        if (value->type == vt_uint) {
            if (value->i > INT8_MAX) {
                error_sym(P, sym, "8-bit signed byte overflow");
                value->type = vt_invalid;
                return -1;
            }
            value->i = (int64_t)value->u;
            value->type = vt_int;
            return 0;
        }
        error_sym(P, sym, "8-bit signed byte type only accepts integers");
        value->type = vt_invalid;
        return -1;
    case fb_bool:
        if (value->type == vt_uint && P->opts.allow_boolean_conversion) {
            if (value->u > 1) {
                error_sym(P, sym, "boolean integer conversion only accepts 0 (false) or 1 (true)");
                value->type = vt_invalid;
                return -1;
            }
        } else if (value->type != vt_bool) {
            error_sym(P, sym, "boolean type only accepts 'true' or 'false' as values");
            value->type = vt_invalid;
            return -1;
        }
        return 0;
    case fb_double:
        switch (value->type) {
        case vt_int:
            d = (double)value->i;
            if ((int64_t)d != value->i) {
                /* We could make this a warning. */
                error_sym(P, sym, "precision loss in 64-bit double type assignment");
                value->type = vt_invalid;
                return -1;
            }
            value->f = d;
            value->type = vt_float;
            return 0;
        case vt_uint:
            d = (double)value->u;
            if ((uint64_t)d != value->u) {
                /* We could make this a warning. */
                error_sym(P, sym, "precision loss in 64-bit double type assignment");
                value->type = vt_invalid;
                return -1;
            }
            value->f = d;
            value->type = vt_float;
            return 0;
        case vt_float:
            /* Double is our internal repr., so not loss at this point. */
            return 0;
        default:
            error_sym(P, sym, "64-bit double type only accepts integer and float values");
            value->type = vt_invalid;
            return -1;
        }
    case fb_float:
        switch (value->type) {
        case vt_int:
            f = (float)value->i;
            if ((int64_t)f != value->i) {
                /* We could make this a warning. */
                error_sym(P, sym, "precision loss in 32-bit float type assignment");
                value->type = vt_invalid;
                return -1;
            }
            value->f = f;
            value->type = vt_float;
            return 0;
        case vt_uint:
            f = (float)value->u;
            if ((uint64_t)f != value->u) {
                /* We could make this a warning. */
                error_sym(P, sym, "precision loss in 32-bit float type assignment");
                value->type = vt_invalid;
                return -1;
            }
            value->f = f;
            value->type = vt_float;
            return 0;
        case vt_float:
            return 0;
        default:
            error_sym(P, sym, "32-bit float type only accepts integer and float values");
            value->type = vt_invalid;
            return -1;
        }
    default:
        error_sym(P, sym, "scalar type expected");
        value->type = vt_invalid;
        return -1;
    }
}

