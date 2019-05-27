import numpy as np

size = 8
bit_num = size * 2

A = 1
B = 3

Q1 = np.zeros((bit_num, bit_num))
C1 = A

for i in range(size):
    q1 = size + i
    Q1[q1][q1] = A
    q2 = i
    Q1[q2][q1] = -2 * A
    for j in range(i + 1, size):
        q2 = j
        Q1[q2][q1] = B

print(Q1)

Q = Q1
C = C1

import wildqat as wq

optimizer = wq.opt()
optimizer.qubo = Q

result = optimizer.sa()

result_mat = np.array([result])

def calc_energy(result_mat, Q, C):
    return (result_mat @ Q @ result_mat.T)[0, 0] + C

print(calc_energy(result_mat, Q1, C1))

def print_bit_row(result_mat, size):
    print_mat = result_mat.reshape((2, size))
    print(print_mat)

print_bit_row(result_mat, size)


Q2 = np.zeros((bit_num, bit_num))

for i in range(size):
    q = size + i
    Q2[q][q] = i + 1

print(Q2)

N = 4

Q3 = np.zeros((bit_num, bit_num))
for q1 in range(size):
    Q3[q1][q1] = -2 * N + 1
    for q2 in range(q1 + 1, size):
        Q3[q1][q2] = 2

C3 = N ** 2

print(Q3)

α = 10
β = 10
γ = 1

Q4 = α * Q1 + β * Q3
C4 = α * C1 + β * C3
Q = Q4 + γ * Q2
C = C4

optimizer = wq.opt()
optimizer.qubo = Q

result = optimizer.sa()

result_mat = np.array([result])

print(calc_energy(result_mat, Q4, C4))
print(calc_energy(result_mat, Q, C))
print_bit_row(result_mat, size)

