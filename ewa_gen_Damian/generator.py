import random

good = ['a', 'b', 'c', 'd', 'e']
neutral = ['f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u']
bad = ['v', 'w', 'x', 'y', 'z']



def slowniki():
    g = open("good.csv", "w")
    for n in good:
        g.write(n)
        g.write('\n')
    g.close()

    b = open("bad.csv", "w")
    for n in bad:
        b.write(n)
        b.write('\n')
    b.close()


def generate_neutral(file):
    dane = []
    dane += (random.choices(bad, k=5))
    dane += (random.choices(good, k=5))
    dane += (random.choices(neutral, k=5))
    random.shuffle(dane)
    file.write(' '.join(dane))
    file.write(',2\n')

def generate_good(file):
    dane = []
    dane += (random.choices(good, k=5))
    dane += (random.choices(neutral, k=10))
    random.shuffle(dane)
    file.write(' '.join(dane))
    file.write(',1\n')

def generate_bad(file):
    dane = []
    dane += (random.choices(bad, k=5))
    dane += (random.choices(neutral, k=10))
    random.shuffle(dane)
    file.write(' '.join(dane))
    file.write(',0\n')

f = open("opinie.csv", "w")

f.write('opinia,sentyment\n')

for n in range(10):
    generate_neutral(f)

for n in range(10):
    generate_good(f)

for n in range(10):
    generate_bad(f)

slowniki()

f.close()
