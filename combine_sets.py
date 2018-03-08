import csv


file1 = open('CAvideos.csv', 'r', encoding='utf8')
file2 = open('GBvideos.csv', 'r', encoding='utf8')
file3 = open('USvideos.csv', 'r', encoding='utf8')

data = []
[data.append(row) for row in csv.reader(file1, delimiter=',')]
[data.append(row) for row in csv.reader(file2, delimiter=',')]
[data.append(row) for row in csv.reader(file3, delimiter=',')]

categories = {}
for i in range(50):
    categories[i] = 0
    
good_data = [data[0]]
ommited = 0
for row in data:
    if row not in good_data:
        try:
            categories[int(row[4])] += 1
            good_data.append(row)
        except:
            print(row)
    else:
        ommited += 1

with open('fullset.csv', 'w', newline='', encoding='utf8') as f:
    writer = csv.writer(f)
    for row in good_data:
        writer.writerow(row)

# prints the number of ommited rows as a sanity check
print(ommited)
