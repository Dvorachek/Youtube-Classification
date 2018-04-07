import csv
from random import shuffle

def main():
    # Balancing from filtered dataset
    file = open('filteredset.csv', 'r', encoding='utf8')
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

    with open('balancedset.csv', 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        for row in balanced_data:
            writer.writerow(row)

if __name__=="__main__":
    main()
