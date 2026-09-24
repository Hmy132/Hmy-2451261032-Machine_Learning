def g(x):
    return (1/3)*x**3 - x

def grad_g(x):
    return x**2 - 1

def gradient_descent_g(x0, eta):
    x = x0

    for i in range(1000):
        x_new = x - eta * grad_g(x)

        if abs(grad_g(x_new)) < 0.001:
            x = x_new
            break

        x = x_new

    return x, g(x), i + 1


x_min, g_min, iterations = gradient_descent_g(2, 0.1)

print("x =", x_min)
print("g(x) =", g_min)
print("So lan lap =", iterations)