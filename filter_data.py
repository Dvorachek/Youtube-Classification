import csv


file = open('fullset.csv', 'r', encoding='utf8')
data = []
[data.append(row) for row in csv.reader(file, delimiter=',')]

categories = {}
for i in range(50):
    categories[i] = 0

for row in data[1:]:
    categories[int(row[4])] += 1

print(categories)
category_count = []
for key, item in categories.items():
    if item:
        category_count.append((key, item))

category_count = sorted(category_count, key=lambda x: x[1], reverse=True)

category_count = category_count[:10]
keep = set()
[keep.add(item[0]) for item in category_count]
print(keep)

# remove duplicates
final_data = [data[0]]
count = 0
for row in data:
    try:
        if int(row[4]) in keep:
            final_data.append(row)
            count += 1
    except:
        pass

print(category_count)
with open('filteredset.csv', 'w', newline='', encoding='utf8') as f:
    writer = csv.writer(f)
    for row in final_data:
        writer.writerow(row)
