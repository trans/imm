"""Cut books.txt into one text per shard, every book preceded and followed by
0x01 (one 0x01 between two books), and write the whole text in book order
the same way (for the check).

usage: python3 -I split_shards.py <books.txt> <books.starts> <books.names>
           <groups.books> <out dir> <whole text out>
           [<extra text> <extra starts> <extra names>]

groups.books is mobius's list: one line per book, name TAB shard TAB chars,
in books.names order, after a header line starting with '#'. A book's bytes
are books.starts[i] to books.starts[i+1] (the last one to the file's end).
Writes <out dir>/shard_NN.txt and prints "key value" lines. An extra text
(with its own starts and names) becomes one more shard, after the listed
ones, and goes into the whole text after the listed books.
"""
import mmap
import os
import sys

SEP = b'\x01'


def read_book_list(text_path, starts_path, names_path):
    size = os.path.getsize(text_path)
    with open(starts_path) as f:
        starts = [int(x) for x in f]
    with open(names_path) as f:
        names = [x.rstrip('\n') for x in f]
    if len(starts) != len(names):
        sys.exit('%s: %d starts, %d names' % (text_path, len(starts), len(names)))
    if starts[0] != 0 or any(b <= a for a, b in zip(starts, starts[1:])) or starts[-1] >= size:
        sys.exit('%s: the starts are not increasing from 0 within the text' % starts_path)
    return size, starts, names


def copy_books(text_path, starts, ends, names, shard_of, outs, whole, books, chars):
    with open(text_path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        for i in range(len(starts)):
            b = m[starts[i]:ends[i]]
            if b'\x00' in b or SEP in b:
                sys.exit('%s: book %d (%s) has a 0x00 or 0x01 byte' % (text_path, i, names[i]))
            s = shard_of[i]
            outs[s].write(b)
            outs[s].write(SEP)
            whole.write(b)
            whole.write(SEP)
            books[s] += 1
            chars[s] += len(b) + 1


def main():
    text_path, starts_path, names_path, groups_path, out_dir, whole_path = sys.argv[1:7]
    extra = sys.argv[7:10] if len(sys.argv) > 7 else None
    size, starts, names = read_book_list(text_path, starts_path, names_path)
    with open(groups_path) as f:
        rows = [x.rstrip('\n').split('\t') for x in f if not x.startswith('#')]
    if len(rows) != len(names):
        sys.exit('lengths differ: %d books, %d groups rows' % (len(names), len(rows)))
    ends = starts[1:] + [size]
    shard_of = []
    for i, (name, shard, chars) in enumerate(rows):
        if name != names[i]:
            sys.exit('book %d: groups has %s, books.names has %s' % (i, name, names[i]))
        if int(chars) != ends[i] - starts[i]:
            sys.exit('book %d (%s): groups says %s chars, the starts say %d' % (i, name, chars, ends[i] - starts[i]))
        shard_of.append(int(shard))
    if min(shard_of) < 0:
        sys.exit('a book has no shard')
    nshards = max(shard_of) + 1
    if extra:
        esize, estarts, enames = read_book_list(*extra)
        nshards += 1

    os.makedirs(out_dir, exist_ok=True)
    outs = [open(os.path.join(out_dir, 'shard_%02d.txt' % s), 'wb') for s in range(nshards)]
    books = [0] * nshards
    chars = [1] * nshards  # the leading separator
    for o in outs:
        o.write(SEP)
    with open(whole_path, 'wb') as whole:
        whole.write(SEP)
        copy_books(text_path, starts, ends, names, shard_of, outs, whole, books, chars)
        nbooks, nchars = len(starts), size
        if extra:
            copy_books(extra[0], estarts, estarts[1:] + [esize], enames, [nshards - 1] * len(estarts),
                       outs, whole, books, chars)
            nbooks, nchars = nbooks + len(estarts), nchars + esize
    for o in outs:
        o.close()

    print('books', nbooks)
    print('text_chars', nchars)
    print('whole_chars', nchars + nbooks + 1)
    print('shards', nshards)
    print('# shard_<i> books chars (separators included)')
    for s in range(nshards):
        print('shard_%02d' % s, books[s], chars[s])
    print('largest_shard_chars', max(chars))


if __name__ == '__main__':
    main()
