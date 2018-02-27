import csv


# file1 = open('CAvideos.csv', 'r', encoding='utf8')
# file2 = open('GBvideos.csv', 'r', encoding='utf8')
# file3 = open('USvideos.csv', 'r', encoding='utf8')
# data = []

# [data.append(row) for row in csv.reader(file1, delimiter=',')]
# [data.append(row) for row in csv.reader(file2, delimiter=',')]
# [data.append(row) for row in csv.reader(file3, delimiter=',')]

categories = {}
for i in range(50):
    categories[i] = 0

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

file = open('fullset.csv', 'r', encoding='utf8')
data = []
[data.append(row) for row in csv.reader(file, delimiter=',')]

for row in data[1:]:
    # try:
    categories[int(row[4])] += 1
    # except:
        

# print(categories)

category_count = []
for key, item in categories.items():
    if item:
        category_count.append((key, item))

category_count = sorted(category_count, key=lambda x: x[1], reverse=True)
category_count = category_count[:10]

keep = set()
[keep.add(item[0]) for item in category_count]
print(keep)

final_data = [data[0]]
count = 0
for row in data:
    try:
        if int(row[4]) in keep:
            final_data.append(row)
            count += 1
    except:
        pass
        
print(count)

with open('filteredset.csv', 'w', newline='', encoding='utf8') as f:
    writer = csv.writer(f)
    for row in final_data:
        writer.writerow(row)



# for item in good_data[-1]:
    # print(item)

    
# print(good_data[-1])
    # data = csv.reader(f, delimiter=' ')

    # i = 1
    # for row in data:
        # print("========================")
        # print(row)
        # print(i)
        # i += 1
        # break
    