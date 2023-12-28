"""
Random order for a sum
======================

Parallelization usually means a summation is done with a random order.
That may lead to different values if the computation is made many times
even though the result should be the same. This example compares
summation of random permutation of the same array of values.

Setup
+++++
"""
from tqdm import tqdm
import numpy as np

unique_values = np.array(
    [2.1102535724639893, 0.5986238718032837, -0.49545818567276], dtype=np.float32
)
random_index = np.random.randint(0, 3, 2000)
assert set(random_index) == {0, 1, 2}
values = unique_values[random_index]

s0 = values.sum()
s1 = np.array(0, dtype=np.float32)
for n in values:
    s1 += n

delta = s1 - s0
print(f"reduced sum={s0}, iterative sum={s1}, delta={delta}")

################################
# There are discrepancies.
#
# Random order
# ++++++++++++
#
# Let's go further and check the sum of random permutation of the same set.
# Let's compare the result with the same sum done with a higher precision (double).


def check_orders(values, n=200, bias=0):
    double_sums = []
    sums = []
    reduced_sums = []
    for i in tqdm(range(n)):
        permuted_values = np.random.permutation(values)
        s = np.array(bias, dtype=np.float32)
        sd = np.array(bias, dtype=np.float64)
        for n in permuted_values:
            s += n
            sd += n
        sums.append(s)
        double_sums.append(sd)
        reduced_sums.append(permuted_values.sum() + bias)

    mi, ma = min(sums), max(sums)
    print(f"min={mi} max={ma} delta={ma-mi}")
    mi, ma = min(double_sums), max(double_sums)
    print(f"min={mi} max={ma} delta={ma-mi} (double)")
    mi, ma = min(reduced_sums), max(reduced_sums)
    print(f"min={mi} max={ma} delta={ma-mi} (reduced)")


check_orders(values)

############################
# This example clearly shows the order has an impact.
# It is usually unavoidable but it could reduced if the sum
# it close to zero. In that case, the sum would be of the same
# order of magnitude of the add values.
#
# Removing the average
# ++++++++++++++++++++
#
# Computing the average of the values requires to compute the sum.
# However if we have an estimator of this average, not necessarily
# the exact value, we would help the summation to keep the same order
# of magnitude than the values it adds.

mean = unique_values.mean()
values -= mean
check_orders(values, bias=len(values) * mean)

######################################
# The differences are clearly lower.
