import csv
from random import shuffle

'''
# file1 = open('CAvideos.csv', 'r', encoding='utf8')
# file2 = open('GBvideos.csv', 'r', encoding='utf8')
# file3 = open('USvideos.csv', 'r', encoding='utf8')
# data = []

# [data.append(row) for row in csv.reader(file1, delimiter=',')]
# [data.append(row) for row in csv.reader(file2, delimiter=',')]
# [data.append(row) for row in csv.reader(file3, delimiter=',')]

# categories = {}
# for i in range(50):
    # categories[i] = 0

# good_data = [data[0]]
# ommited = 0
# for row in data:
    # if row not in good_data:
        # try:
            # categories[int(row[4])] += 1
            # good_data.append(row)
        # except:
            # print(row)
    # else:
        # ommited += 1

# with open('fullset.csv', 'w', newline='', encoding='utf8') as f:
    # writer = csv.writer(f)
    # for row in good_data:
        # writer.writerow(row)

# print(ommited)

# file = open('fullset.csv', 'r', encoding='utf8')
# data = []
# [data.append(row) for row in csv.reader(file, delimiter=',')]

# for row in data[1:]:
    # categories[int(row[4])] += 1


# print(categories)

# category_count = []
# for key, item in categories.items():
    # if item:
        # category_count.append((key, item))

# category_count = sorted(category_count, key=lambda x: x[1], reverse=True)
# category_count = category_count[:10]

# keep = set()
# [keep.add(item[0]) for item in category_count]
# print(keep)

# remove duplicates
# final_data = [data[0]]
# count = 0
# for row in data:
    # try:
        # if int(row[4]) in keep:
            # final_data.append(row)
            # count += 1
    # except:
        # pass
        
# print(category_count)

# with open('filteredset.csv', 'w', newline='', encoding='utf8') as f:
    # writer = csv.writer(f)
    # for row in final_data:
        # writer.writerow(row)
'''

# Balancing from filtered dataset
file = open('filteredalphaset.csv', 'r', encoding='utf8')
data = []
[data.append(row) for row in csv.reader(file, delimiter=',')]


categories = {}
for i in range(50):
    categories[i] = 0

for row in data[1:]:
    categories[int(row[4])] += 1
    
category_count = []
for key, item in categories.items():
    if item:
        category_count.append((key, item))

category_count = sorted(category_count, key=lambda x: x[1], reverse=True)
category_count = category_count[:10]

keep = set()
[keep.add(item[0]) for item in category_count]
print(keep)

li = [value[1] for value in category_count]
avg = sum(li)/len(li)

header = data[0]
payload = data[1:]

shuffle(payload)

balanced_data = [header]

for i in range(50):
    categories[i] = 0

for row in payload:
    if categories[int(row[4])] < avg:
        balanced_data.append(row)
        categories[int(row[4])] += 1
    

print(categories)


with open('balancedalphaset.csv', 'w', newline='', encoding='utf8') as f:
    writer = csv.writer(f)
    for row in balanced_data:
        writer.writerow(row)

