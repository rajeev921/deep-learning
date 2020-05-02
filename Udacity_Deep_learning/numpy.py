import numpy as np

#Data Manipulation
x = np.arange(12)
x
# array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.])

x.shape
#(12,)   ndarray’s shape (the length along each axis)

x.size
#12      the total number of elements in an ndarray, i.e., the product of all of the shape elements, 

x = x.reshape(3, 4)
x
'''
array([[ 0.,  1.,  2.,  3.],
       [ 4.,  5.,  6.,  7.],
       [ 8.,  9., 10., 11.]])
'''

# The empty method grabs a chunk of memory and hands us back a matrix without bothering to change the value of any of its entries.

np.empty((3, 4))

'''
array([[-8.5251129e+17,  4.5683731e-41, -2.3859901e+00,  3.0879013e-41],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00]])
'''

# our matrices initialized either with zeros, ones, some other constants, or numbers randomly sampled from a specific distribution

np.zeros((2, 3, 4))

'''
array([[[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]],

       [[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]]])
'''

# we can create tensors with each element set to 1

np.ones((2, 3, 4))

'''
array([[[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]],

       [[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]]])
'''

# Randomly sample the values for each element. Each of its elements is randomly sampled from a standard Gaussian (normal) distribution with a mean of 0 and a standard deviation of 1.

np.random.normal(0, 1, size=(3, 4))
'''
array([[ 2.2122064 ,  1.1630787 ,  0.7740038 ,  0.4838046 ],
       [ 1.0434405 ,  0.29956347,  1.1839255 ,  0.15302546],
       [ 1.8917114 , -1.1688148 , -1.2347414 ,  1.5580711 ]])
'''

# we initialize weights by sampling random numbers from a normal distribution with mean 0 and a standard deviation of  0.01 , setting the bias  b  to  0 .

w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)


# We can also specify the exact values for each element in the desired ndarray by supplying a Python list (or list of lists) containing the numerical values. Here, the outermost list corresponds to axis  0 , and the inner list to axis  1 .

np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
'''
array([[2., 1., 4., 3.],
       [1., 2., 3., 4.],
       [4., 3., 2., 1.]])
'''
################ Operations ####################
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])

x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
'''
(array([ 3.,  4.,  6., 10.]),
 array([-1.,  0.,  2.,  6.]),
 array([ 2.,  4.,  8., 16.]),
 array([0.5, 1. , 2. , 4. ]),
 array([ 1.,  4., 16., 64.]))
'''

np.exp(x)
# array([2.7182817e+00, 7.3890562e+00, 5.4598148e+01, 2.9809580e+03])

# numpy concatenate
x = np.arange(12).reshape(3, 4)
y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

np.concatenate([x, y], axis=0), np.concatenate([x, y], axis=1)

'''
(array([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [ 2.,  1.,  4.,  3.],
        [ 1.,  2.,  3.,  4.],
        [ 4.,  3.,  2.,  1.]]),

 array([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
        [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
        [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]]))
'''
x == y
'''
array([[False,  True, False,  True],
       [False, False, False, False],
       [False, False, False, False]])
'''
x.sum()
# array(66.)


################## Broadcasting ##################
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b

'''
(array([[0.],
        [1.],
        [2.]]),
 array([[0., 1.]]))
'''

a + b
'''
array([[0., 1.],
       [1., 2.],
       [2., 3.]])
'''

####################### Indexing and Slicing #######################
x[-1], x[1:3]  # select 2nd and 3rd element

'''
(array([ 8.,  9., 10., 11.]),
 array([[ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]]))
'''

x[1, 2] = 9
x

'''
array([[ 0.,  1.,  2.,  3.],
       [ 4.,  5.,  9.,  7.],
       [ 8.,  9., 10., 11.]])
'''

# If we want to assign multiple elements the same value, we simply index all of them and then assign them the value. For instance, [0:2, :] accesses the first and second rows, where : takes all the elements along axis  1  (column). 

x[0:2, :] = 12
x

'''
array([[12., 12., 12., 12.],
       [12., 12., 12., 12.],
       [ 8.,  9., 10., 11.]])
'''       

#Data Preprocessing

####################### Handling Missing Data #######################

# Note that “NaN” entries are missing values. To handle missing data, typical methods include imputation and deletion, where imputation replaces missing values with substituted ones, while deletion ignores missing values.

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())

print(inputs)

'''
   NumRooms Alley
0       3.0  Pave
1       2.0   NaN
2       4.0   NaN
3       3.0   NaN
'''

'''
For categorical or discrete values in inputs, we consider “NaN” as a category. Since the “Alley” column only takes 2 types of categorical values “Pave” and “NaN”, pandas can automatically convert this column to 2 columns “Alley_Pave” and “Alley_nan”. A row whose alley type is “Pave” will set values of “Alley_Pave” and “Alley_nan” to  1  and  0 . A row with a missing alley type will set their values to  0  and  1 .
'''

inputs = pd.get_dummies(inputs, dummy_na=True)

print(inputs)

'''
   NumRooms  Alley_Pave  Alley_nan
0       3.0           1          0
1       2.0           0          1
2       4.0           0          1
3       3.0           0          1
'''

#Linear Algebra
'''   Norms
Some of the most useful operators in linear algebra are norms. Informally, the norm of a vector tells us how big a vector is. The notion of size under consideration here concerns not dimensionality but rather the magnitude of the components.
In linear algebra, a vector norm is a function f that maps a vector to a scalar, satisfying a handful of properties. Given any vector x, the first property says that if we scale all the elements of a vector by a constant factor α, its norm also scales by the absolute value of the same constant factor:
'''

f(αx)=|α|f(x)


#Calculus


#Automatic Differentiation


#Probability
# In statistics we call this process of drawing examples from probability distributions sampling. 
# The distribution that assigns probabilities to a number of discrete choices is called the multinomial distribution. 

fair_probs = [1.0 / 6] * 6

np.random.multinomial(1, fair_probs)