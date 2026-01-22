print('chocho')


print("Hello, Jigsaw!")
def greet():
    return "Greetings from Jigsaw!"

if __name__ == "__main__":
    print(greet())
def add(a, b):
    return a + b
def subtract(a, b):
    return a - b
def multiply(a, b):
    return a * b
def divide(a, b):
    if b == 0:
        return "Error: Division by zero"
    return a / b
def power(a, b):
    return a ** b
def modulus(a, b):
    return a % b
def floor_divide(a, b):
    if b == 0:
        return "Error: Division by zero"
    return a // b
def square_root(a):
    if a < 0:
        return "Error: Negative input"
    return a ** 0.5
def cube_root(a):
    if a < 0:
        return -(-a) ** (1/3)
    return a ** (1/3)
def factorial(n):
    if n < 0:
        return "Error: Negative input"
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
def fibonacci(n):
    