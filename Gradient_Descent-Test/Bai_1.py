def f(x):
    return x**2 - 2

def grad_f(x):
    return 2*x

def gradient_descent_f(x0, eta):
    x = x0

    for i in range(1000):
        x_new = x - eta * grad_f(x)

        if abs(grad_f(x_new)) < 0.001:
            x = x_new
            break

        x = x_new

    return x, f(x), i + 1


x_min, f_min, iterations = gradient_descent_f(5, 0.1)

print("x =", x_min)
print("f(x) =", f_min)
print("So lan lap =", iterations)