import numpy

C = numpy.array([
    [1, -1, 0],
    [0, 1, 0],
    [0, 0, 1]
])

A = numpy.eye(C.shape[0])
A_inv = numpy.eye(C.shape[0])

columns = list(enumerate(zip(*C)))
perms = []

for i in xrange(C.shape[1]):
    l = numpy.dot(A_inv, columns[0][1])
    k = 0
    while l[i] == 0:
        k += 1
        l = numpy.dot(A_inv, columns[k][1])

    perms.append((columns[k][0], i))

    A[:, i] = columns[k][1]
    del columns[k]
    l_tilda = numpy.array(l)
    l_tilda[i] = -1
    q = numpy.dot(-1.0 / l[i], l_tilda)
    Q = numpy.eye(C.shape[0])
    Q[:, i] = numpy.array(q)
    A_inv = numpy.dot(Q, A_inv)

result = numpy.eye(C.shape[0])
for perm in perms:
    result[perm[0], :] = A_inv[perm[1], :]

print "Inverted matrix:"
print result
print

print "Check result by build-in function:"
print numpy.linalg.inv(C)
