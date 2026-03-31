a = [(1, 2), (100, 1000), (1, 1 )]

b =zip(*a)

for x, y, z in b:
    print(x, y, z)
