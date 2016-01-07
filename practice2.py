__author__ = 'jt306'
import numpy.random as random
random_bits = 0
for i in range(64):
    if random.randint(0,1):
        print (True)
        random_bits |= 1 <<i
print (random_bits)


def sort_priority(values,group):
    def helper(x):
        if x in group:
            return(0,x)
        return (1,x)
    values.sort(key=helper)


numbers=[8,3,1,2,4,5,7,6]
group = {7,2,5,3,7}
print( sort_priority(numbers,group))
print(group)
print(numbers)

def generate_power_func(n):
    print ("id(n): %X" % id(n))
    def nth_power(x):
        return x**n
    print ("id(nth_power): %X" % id(nth_power))
    return nth_power

generate_power_func(5)