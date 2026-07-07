# List Methods Example

numbers = [10, 20, 30]
print("Original List:", numbers)

# append()
numbers.append(40)
print("After append:", numbers)

# insert()
numbers.insert(1, 15)
print("After insert:", numbers)

# extend()
numbers.extend([50, 60])
print("After extend:", numbers)

# remove()
numbers.remove(20)
print("After remove:", numbers)

# pop()
numbers.pop()
print("After pop:", numbers)

# sort()
numbers.sort()
print("After sort:", numbers)

# reverse()
numbers.reverse()
print("After reverse:", numbers)

# index()
print("Index of 30:", numbers.index(30))

# count()
print("Count of 10:", numbers.count(10))