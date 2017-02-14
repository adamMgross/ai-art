import json

f = open('./data.json', 'r')
data = json.load(f)
first = data[0]
print first, type(first)

f.close()
