import csv

file = open('fullset.csv', 'r', encoding='utf8')

data = [row for row in csv.reader(file, delimiter=',')]

file.close()

categories = {}
for i in range(50):
    categories[i] = 0

better_data = [[row[4], row[2], row[6], row[15]] for row in data]
#print(better_data)

good_data = [better_data[0]]
ommited = 0
for row in better_data:
    if row not in good_data:
       # print(row)
        try:
            categories[int(row[0])] += 1
            good_data.append(row)
        except:
            pass
            #print(row)
    else:
        ommited += 1
#print(good_data)

with open('fullset2.csv', 'w', newline='', encoding='utf8') as f:
    writer = csv.writer(f)
    for row in good_data:
        writer.writerow(row)

#print(ommited)

