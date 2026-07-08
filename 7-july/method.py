#1)Dictionary


student = {
    "name": "Rahul",
    "age": 20
}

# get()
print(student.get("name"))

# keys()
print(student.keys())

# values()
print(student.values())

# update()
student.update({"city": "Surat"})
print(student)

# pop()
student.pop("age")
print(student)


#2)list

fruits = ["Apple", "Banana"]

# append()
fruits.append("Mango")

# insert()
fruits.insert(1, "Orange")

# remove()
fruits.remove("Banana")

# pop()
fruits.pop()

# sort()
fruits.sort()

print(fruits)

#3)TUPLE


numbers = (10, 20, 30, 20, 40)

# count()
print(numbers.count(20))

# index()
print(numbers.index(30))

# len()
print(len(numbers))

# max()
print(max(numbers))

# min()
print(min(numbers))


#4)set

myset = {10, 20, 30}

# add()
myset.add(40)

# remove()
myset.remove(20)

# discard()
myset.discard(50)

# pop()
myset.pop()

print(myset)

# clear()
myset.clear()

print(myset)

