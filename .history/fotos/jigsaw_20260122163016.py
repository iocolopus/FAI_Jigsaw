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
    if n < 0:
        return "Error: Negative input"
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
def lcm(a, b):
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)
def average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
def median(numbers):
    if not numbers:
        return 0
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2
    else:
        return sorted_numbers[mid]
    