import csv

file = open('filteredset.csv', 'r', encoding='utf8')
data = []
[data.append(row) for row in csv.reader(file, delimiter=',')]

alpha_data = [data[1]]
for row in data[1:]:
    li = row[15].split(' ')
    only_words = [token for token in li if token.isalpha()]
    row[15] = ' '.join(only_words)
    
    alpha_data.append(row)
    
with open('filteredalphaset.csv', 'w', encoding='utf8') as f:
    writer = csv.writer(f)
    for row in alpha_data:
        writer.writerow(row)

