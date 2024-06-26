import numpy as np
m = np.array([[1, 2, 3],
			[2, 3, 4],
			[4, 5, 6]])
print("Printing the Original square array:\n",m)
print()
print('***************************************')
print()
w, v = np.linalg.eig(m)
print("Printing the Eigen values of the given square array:\n",w)
print()
print("Printing Right Eigen Vectors of the given square array:\n",v)

!pip install tensorflow

!pip install tensorflow[and-cuda]

import tensorflow as tf

e_matrix_A = tf.random.uniform([2, 2], minval=3, maxval=10, dtype=tf.float32, name="matrixA")
print("Matrix A: \n{}\n\n".format(e_matrix_A))

eigen_values_A, eigen_vectors_A = tf.linalg.eigh(e_matrix_A)
print("Eigen Vectors: \n{} \n\nEigen Values: \n{}\n".format(eigen_vectors_A, eigen_values_A))

e_matrix_A = tf.random.uniform([3, 3], minval=3, maxval=10, dtype=tf.float32, name="matrixA")
print("Matrix A: \n{}\n\n".format(e_matrix_A))

eigen_values_A, eigen_vectors_A = tf.linalg.eigh(e_matrix_A)
print("Eigen Vectors: \n{} \n\nEigen Values: \n{}\n".format(eigen_vectors_A, eigen_values_A))