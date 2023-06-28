def add_and_sub(a,b):
    add = a+b
    sub = a-b
    return add, sub

a = 3
b = 4
c, d = add_and_sub(a,b)

print(a,b,c,d)
import random
def set_list(*args, opt="random"):
    # alist = [a for a in args]
    alist = []
    if len(args) != 0:
        for a in args:
            alist.append(a)
    else:
        if opt == "random":
            for j in range(5):
                alist.append(random.random())
        elif opt == "zero":
            alist = [0] * 5
    return alist

a = set_list()
print(a)

add = lambda a, b : a+b

result = add(3,5)
print(result)