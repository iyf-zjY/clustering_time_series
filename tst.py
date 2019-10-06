c = 0
for a in range(1,1000001):
    f1 = 0
    f2 = 0
    f3 = 0
    f4 = 0
    aa = str(a)
    for t in aa:
        if t == '1':
            f1 = 1
    if  not f1:
        c += 1
print(c)