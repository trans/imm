"""Cut books.txt into one text per shard, every book preceded and followed by
0x01 (one 0x01 between two books), and write the whole text in book order
the same way (for the check).

usage: python3 -I split_shards.py <books.txt> <books.starts> <books.names>
           <groups.books> <out dir> <whole text out>

groups.books is mobius's list: one line per book, name TAB shard TAB chars,
in books.names order, after a header line starting with '#'. A book's bytes
are books.starts[i] to books.starts[i+1] (the last one to the file's end).
Writes <out dir>/shard_NN.txt and prints "key value" lines.
"""
import mmap
import os
import sys

SEP = b'\x01'


def main():
    text_path, starts_path, names_path, groups_path, out_dir, whole_path = sys.argv[1:7]
    size = os.path.getsize(text_path)
    with open(starts_path) as f:
        starts = [int(x) for x in f]
    with open(names_path) as f:
        names = [x.rstrip('\n') for x in f]
    with open(groups_path) as f:
        rows = [x.rstrip('\n').split('\t') for x in f if not x.startswith('#')]
    if not (len(starts) == len(names) == len(rows)):
        sys.exit('lengths differ: %d starts, %d names, %d groups rows' % (len(starts), len(names), len(rows)))
    if starts[0] != 0 or any(b <= a for a, b in zip(starts, starts[1:])) or starts[-1] >= size:
        sys.exit('books.starts is not increasing from 0 within the text')
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

    os.makedirs(out_dir, exist_ok=True)
    outs = [open(os.path.join(out_dir, 'shard_%02d.txt' % s), 'wb') for s in range(nshards)]
    books = [0] * nshards
    chars = [1] * nshards  # the leading separator
    for o in outs:
        o.write(SEP)
    with open(text_path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m, \
            open(whole_path, 'wb') as whole:
        whole.write(SEP)
        for i in range(len(starts)):
            b = m[starts[i]:ends[i]]
            if b'\x00' in b or SEP in b:
                sys.exit('book %d (%s) has a 0x00 or 0x01 byte' % (i, names[i]))
            s = shard_of[i]
            outs[s].write(b)
            outs[s].write(SEP)
            whole.write(b)
            whole.write(SEP)
            books[s] += 1
            chars[s] += len(b) + 1
    for o in outs:
        o.close()

    print('books', len(starts))
    print('text_chars', size)
    print('whole_chars', size + len(starts) + 1)
    print('shards', nshards)
    print('# shard_<i> books chars (separators included)')
    for s in range(nshards):
        print('shard_%02d' % s, books[s], chars[s])
    print('largest_shard_chars', max(chars))


if __name__ == '__main__':
    main()
