# if statement
age = 20

if age >= 18:
    print("You are eligible to vote")

# if-else statement
num = 5

if num % 2 == 0:
    print("Even Number")
else:
    print("Odd Number")

# if-elif-else statement
marks = 75

if marks >= 90:
    print("Grade A")
elif marks >= 60:
    print("Grade B")
else:
    print("Grade C")

# Nested if-else
username = "admin"
password = "1234"

if username == "admin":
    if password == "1234":
        print("Login Successful")
    else:
        print("Wrong Password")
else:
    print("Invalid Username")