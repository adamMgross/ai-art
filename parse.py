import json
import time


def do_stuff(chunk, num):
    entry = {}
    for i,line in enumerate(chunk):
        key = line[:line.index(':')]
        val = line[-2] if ',' in line else line[-1]
        if i == 0:
            val = line[line.index(':') + 2:]
        entry[key] = val
    entry['genre'] = 'surrealist' if num < 34 else 'impressionist' if num < 67 else 'neither'

    return entry

f = open('./assignment3.txt', 'r').read().splitlines()[11:]

data = []
i = 0
chunk_num = 0
while i < len(f):
    if '---' in f[i]:
        i += 1
    chunk = f[i:i+11]
    chunk_num += 1
    print chunk
    entry = do_stuff(chunk, chunk_num)
    data.append(entry)
    i += 11
with open('./data.json', 'wr') as outfile:
    json.dump(data, outfile)
print len(data)
