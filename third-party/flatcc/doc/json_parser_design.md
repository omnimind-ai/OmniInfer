# JSON Parser Design

The overall principle of the json parser is as follows:

`flatcc/flatcc_json.h` contains functions to parse json primitives
and type conversions, and a generic json parser to skip unrecognized
fields.

For each table all known fields are sorted and a trie is constructed.
After recognizing a json object, each member name is read partially
and matched against the trie, and more of the field name is read as
demanded by the trie and a match or rejection is decided.

The member name is read as a single 64 bit word read in big endian
format (zero padded at the end of file). Big endian makes the first
character in the name most significant and allows the word to contain
trailing "garbage". The efficiency depends on unaligned 64-bit reads
with a fast byteswapping operation, but there is a fall-back that
works in all cases via an unaligned read support function.

The trie is constructed by sorting all field names and taking a
median split. If the split name has preceeding name which is a strict
prefix, that name is chosen as split instead. This is important
because words are read with trailing "garbage". If the median name is
longer than a 64 bit word, it is treated specially where a match
triggers reading more data and repeating the process.

The median is tested for less than, but not equality. Each side if
the choice uses a new median like above, until there is only one
option left. This option is then tested for exact match by masking
out any trailing data. If the match succeeds the input member name
must be checked for closing `"` or other valid termination if
allowing unquoted names.

Note that this match only visits data once, unlike a hash table where
a lookup still requires matching the name. Since we expect the number
of field names per table to be reasonable, this approach is likely
faster than a hash table.

If the match fails, then an error can be generated with a "unknown
field" error, or the field can be consumed by the generic json
parser.

When a member name is successfully matched againts a known field, the
member value is expected to be of a primitive type such as a string
or an integer, the value is parsed using a json support parser and
type converter which may issue overflow and format errors. For
example, integers won't accept decimal points and ubyte fields won't
accept integers of value 256 or higher. If the parse was successful
the member is added the current flatbuffer table opened at start of
the json object. In fact, the value is parsed directly into the
flatbuffer builders entry for the field.

If a scalar field fails to parse its value, a second attempt is
done parsing the value as a symbolic value. This parse is similar
to parsing member names but uses a global set of known constants
from enumerations. If this also fails, the field is invalid and
an error is generated.

When the field is parsed, comma is detected and the process is
repeated. It may fail on duplicate fields because the builder will
complain.

Once a closing bracket is matched, the table is closed in the
builder.

In the above we discussed tables, but structs work the same with the
exception that any fields not present are simply set to zero and it
is not checked. Nor are duplicate fields checked. This is to avoid
allocating extra space for something that isn't very important.

Complex member types require more consideration. We wish to avoid a
recursive descent parser because it imposes limits on nesting depth,
but we want to have a composable parser. This leads to solutions such
as a trampoline, returning a parser function for the matched field
which is called by a driver function, before resuming the current
parser. We wish to avoid this as it adds overhead, especially in
inlining.

To avoid trampolines we compile all tables and structs into one large
function where each type can be reached by a goto label or a switch
statement. We can then use a variant of Simon Tathams famous
co-routine by duff device to abort and resume the current parse. We
still need a stack to track return state.
<http://www.chiark.greenend.org.uk/~sgtatham/coroutines.html>

Outside the giant state machine we provide wrappers so we can parse
each type from a clean call interface to parse root type.

The above solution has one major problem: we apparently have to
include types from included schema as well. But we don't have to it
turns out: Each included schema is closed and can never have a member
type in our current schema. Therefore mutual recursion is not
possible. This means we can afford to use function call recursion
into included parsers with recursion depth bounded by the number of
included schema.

Spaces are optimized by first testing to see if there is a character
above space and exit immediately if so.  Otherwise, if unaligned
access is valid and the buffer holds at least 16 bytes, we descend
hierarchically to test ever small unligned loads against a word of
spaces. This loops as long as 2x64 bit words can be matched to a word
of pure spaces (ASCII 0x20).  If there is any control character, or
if we reach near the end or, or if unaligned access is not available,
we bail out to a standard character parsing loop that also does line
counting.

A final, and major issue, is how to handle unions:

Googles `flatc` compiler makes the assumtion that the union type
field appear before the union table field. However, this order cannot
be trusted in standard JSON. If we were to sort the fields first
or prescan for type, it would complete ruin our parsing strategy.

To handle unions efficiently we therefore require either that union
fields appear before the union table, and that the associated
union table is present before another unions type field, or we
require that the member name is tagged with its type. We prefer
the tagged approach, but it is not compliant with `flatc v1.2`.

By using a tag such as `"test_as_Monster"` instead of just "test",
the parser can treat the field as any other table field. Of course
there is only allowed to be one instance, so `"test_as_Weapon"`
in the same object would not be allowed. To the parser it just
triggers a duplicate field error.

In addition, the `"test_type"` field is allowed to appear anywhere,
or to be absent. If it is present, it must have a type that
matches the tagged member, unless it is of type NONE. NONE may
also be represented as `"test_as_NONE": null`.

While the tagged approach has no use of the type field, it may still
be very useful to other JSON consumers so they can know what tagged
field to look up in the JSON object.

The parser handles the type field by setting the type field in the
flatcc builder table and checking if it is already set when
seeing a tagged or untagged union table field.
