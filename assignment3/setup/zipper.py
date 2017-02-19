import json
import random
with open('../data.json', 'r') as infile:
    lst = json.load(infile)
    random.shuffle(lst)
    with open('./zipped/training_set.json', 'w') as trs:
        json.dump(lst[:80], trs)
    with open('./zipped/test_set.json', 'w') as tsts:
        json.dump(lst[80:], tsts)
