def fib(b):
    n = int(b)
    if n == 0: 
        return 0
    if n == 1: 
        return 1
    return fib(n-2) + fib(n-1)

b = os.getenv('b')
print(fib(b))
