import chained_operations as op
import optimizers
import numpy as np

a = op.Placeholder()
b = op.Placeholder()

g_a = op.Gradient(a)
g_b = op.Gradient(b)

y = op.Mul(-1, b)

data = np.array([1.0, 2.0, 3.0])

out, grad_a, grad_b = op.run([y, g_a, g_b], {a: 2, b: 3})

print(grad_a, grad_b)

# print(optimizers.numeric_gradient(data, lambda inp: op.run(y, {x: inp})))
