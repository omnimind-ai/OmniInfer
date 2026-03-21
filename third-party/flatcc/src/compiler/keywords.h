/*
 * FlatBuffers keyword table
 *
 * See luthor project test files for more details on keyword table
 * syntax.
 *
 * In brief: Keywords are assigned a hash key that is easy
 * for the lexer to test.
 *
 * The first char is length of keyword, two next chars are the leading
 * to characters of the keyword, and the last char is the last char of
 * the keyword. For keywords longer than 9 add length to '0' in the
 * first character. For keywords shorter than 3 characters, see luthor
 * project - we don't need it. The keywords should be sorted.
 */

LEX_KW_TABLE_BEGIN
    lex_kw(int, '3', 'i', 'n', 't')
    lex_kw(bool, '4', 'b', 'o', 'l')
    lex_kw(byte, '4', 'b', 'y', 'e')
    lex_kw(char, '4', 'c', 'h', 'r')
    lex_kw(enum, '4', 'e', 'n', 'm')
    lex_kw(int8, '4', 'i', 'n', '8')
    lex_kw(long, '4', 'l', 'o', 'g')
    lex_kw(null, '4', 'n', 'u', 'l')
    lex_kw(true, '4', 't', 'r', 'e')
    lex_kw(uint, '4', 'u', 'i', 't')
    lex_kw(false, '5', 'f', 'a', 'e')
    lex_kw(float, '5', 'f', 'l', 't')
    lex_kw(int32, '5', 'i', 'n', '2')
    lex_kw(int16, '5', 'i', 'n', '6')
    lex_kw(int64, '5', 'i', 'n', '4')
    lex_kw(table, '5', 't', 'a', 'e')
    lex_kw(ubyte, '5', 'u', 'b', 'e')
    lex_kw(uint8, '5', 'u', 'i', '8')
    lex_kw(ulong, '5', 'u', 'l', 'g')
    lex_kw(union, '5', 'u', 'n', 'n')
    lex_kw(short, '5', 's', 'h', 't')
    lex_kw(double, '6', 'd', 'o', 'e')
    lex_kw(string, '6', 's', 't', 'g')
    lex_kw(struct, '6', 's', 't', 't')
    lex_kw(uint32, '6', 'u', 'i', '2')
    lex_kw(uint16, '6', 'u', 'i', '6')
    lex_kw(uint64, '6', 'u', 'i', '4')
    lex_kw(ushort, '6', 'u', 's', 't')
    lex_kw(float32, '7', 'f', 'l', '2')
    lex_kw(float64, '7', 'f', 'l', '4')
    lex_kw(include, '7', 'i', 'n', 'e')
    lex_kw(attribute, '9', 'a', 't', 'e')
    lex_kw(namespace, '9', 'n', 'a', 'e')
    lex_kw(root_type, '9', 'r', 'o', 'e')
    lex_kw(rpc_service, '0' + 11, 'r', 'p', 'e')
    lex_kw(file_extension, '0' + 14, 'f', 'i', 'n')
    lex_kw(file_identifier, '0' + 15, 'f', 'i', 'r')
LEX_KW_TABLE_END

