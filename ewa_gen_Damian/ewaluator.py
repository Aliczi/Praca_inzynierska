import pandas as pd

f = open("bad.csv", "r")
bad = []
for line in f.readlines():
    # print([line.strip()])
    bad.append(line.strip())
f.close()

f = open("good.csv", "r")
good = []
for line in f.readlines():
    # print([line.strip()])
    good.append(line.strip())
f.close()

df = pd.read_csv('opinie.csv')

poprawne = 0
lacznie = 0

for index, row in df.iterrows():

    dobry, zly = 0, 0
    slowa = row['opinia'].split()
    sentyment_prawdziwy = row['sentyment']

    for n in slowa:
        if n in good:
            dobry += 1
        if n in bad:
            zly += 1

    if dobry == zly:
        sentyment_wykryty = 2
        print('neutralny')
    elif dobry > zly:
        sentyment_wykryty = 1
        print('dobry')
    else:
        sentyment_wykryty = 0
        print('zly')

    if sentyment_prawdziwy == sentyment_wykryty: poprawne += 1

    lacznie += 1

print(poprawne, '/', lacznie)
print(poprawne / lacznie * 100, "% poprawnie przypisanych")
# print(slowa, sentyment, dobry, zly)
