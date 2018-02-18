import types
import sys
import random
import operator
import unittest
from sympy import *
from sympy import Matrix as Mat
import copy
import math

class MatrixError(Exception):
	pass

class MatrixArithmeticError(MatrixError):
	def __init__(self, elementLeft, elementRight, operation):
		self.elementLeft = elementLeft
		self.elementRight = elementRight
		self.operation = operation

	def __str__(self):
		return "Cannot %s a %dx%d and a %dx%d matrix" % (self.operation, self.elementLeft.rows(), self.elementLeft.cols(), self.elementRight.rows(), self.elementRight.cols())

class MatrixAdditionError(MatrixArithmeticError):
	def __init__(self, elementLeft, elementRight):
		MatrixArithmeticError.__init__(self, elementLeft, elementRight, "add")

class MatrixMultiplicationError(MatrixArithmeticError):
	def __init__(self, elementLeft, elementRight):
		MatrixArithmeticError.__init__(self, elementLeft, elementRight, "multiply")

class SquareError(MatrixError):
	def __init__(self, function):
		self.function = function

	def __str__(self):
		return "The %s function is only defined for square matricies." % self.function

class DeterminantError(SquareError):
	def __init__(self):
		SquareError.__init__(self, "determinant")

class InverseError(SquareError):
	def __init__(self):
		SquareError.__init__(self, "inverse")

class FormError(MatrixError):
	def __init__(self, msg):
		self.msg = msg

	def __str__(self):
		return "%s" % self.msg
		
class Matrix:
	def __init__(self, *args):
		if len(args) == 2:
			if isinstance(args[0], types.IntType) and isinstance(args[1], types.IntType):
				self.zeros(args[0], args[1])
			else:
				raise TypeError("Only two integer arguments are accepted.")
		elif len(args) == 1:
			if isinstance(args[0], types.IntType):
				self.zeros(args[0], args[0])
			elif isinstance(args[0], types.ListType):
				self.matrix = args[0]
			else:
				raise TypeError("Only an integer or a list is accepted for one argument.")
		else:
			raise TypeError("Only 1 or 2 arguments are accepted (%d given).") % len(args)
				
	def __str__(self):
		s = ""
		for row in self.matrix:
			s += "%s\n" % row
		return s

	def __getitem__(self, (row, col)):
		return self.matrix[row][col]

	def __setitem__(self, (row, col), value):
		self.matrix[row][col] = value
		
	def __add__(self, other):
		if not isinstance(other, Matrix):
			raise TypeError("Cannot add a matrix and a %s" % type(other))
		if not (self.cols() == other.cols() and self.rows() == other.rows()):
			raise MatrixAdditionError(self, other)
		r = []
		for row in xrange(self.rows()):
			r.append([])
			for col in xrange(self.cols()):
				r[row].append(self[(row, col)] + other[(row, col)])
		return Matrix(r)

	def __sub__(self, other):
		return self + -other

	def __neg__(self):
		return -1 * self

	def __mul__(self, other):
		if self.is_scalar_element(other):
			return self.scalar_multiply(other)
		if  isinstance(other, Matrix):
			return self.matrix_multiply(other)
		elif isinstance(other, tuple):
			return self.tuple_multiply(other)
		else:
			raise TypeError("Cannot multiply matrix and type %s" % type(other))

	def __rmul__(self, other):
		if not self.is_scalar_element(other):
			raise TypeError("Cannot right-multiply by %s" % type(other))
		return self.scalar_multiply(other)

	def __eq__(self, other):
		if not isinstance(other, Matrix):
			raise TypeError("Cannot equal a matrix and a %s" % type(other))
		return all(self.row(i) == other.row(i) for i in xrange(self.rows()))

	def scalar_multiply(self, scalar):
		rows = []
		for row in self.matrix:				
			rows.append(map(lambda x: x * scalar, row))
		return Matrix(rows)

	def matrix_multiply(self, other):
		r = []
		if not isinstance(other, Matrix):
			raise TypeError("Cannot multiply matrix and type %s" % type(other))
		if not self.cols() == other.rows():
			raise MatrixMultiplicationError(self, other)
		for row in xrange(self.rows()):
			r.append([])
			for col in xrange(other.cols()):
				r[row].append(self.vector_inner_product(self.row(row), other.col(col)))
		if len(r) == 1 and len(r[0]) == 1:
			return r[0][0]
		else:
			return Matrix(r)

	def tuple_multiply(self, other):
		r = []
		if not isinstance(other, tuple):
			raise TypeError("Cannot multiply matrix and type %s" % type(other))
		if not self.cols() == len(other):
			raise MatrixMultiplicationError(self, other)
		for row in xrange(self.rows()):
			r.append([])
			r[row].append(self.vector_inner_product(self.row(row), other))
		if len(r) == 1 and len(r[0]) == 1:
			return r[0][0]
		else:
			return Matrix(r)
			
	def vector_inner_product(self, a, b):
		if not isinstance(a, types.ListType):
			raise TypeError("Only two lists are accepted.")
		if not isinstance(b, types.ListType) and not isinstance(b, types.TupleType):
			raise TypeError("Only two lists are accepted.")
		return reduce(operator.add, map(operator.mul, a, b))

	def is_scalar_element(self, x):
		return isinstance(x, types.IntType) or isinstance(x, types.FloatType) or isinstance(x, types.ComplexType)

	def is_row_vector(self):
		return self.rows() == 1 and self.cols() > 1

	def is_column_vector(self):
		return self.cols() == 1 and self.rows() > 1
	
	def row(self, i):
		return self.matrix[i]
		
	def col(self, j):
		r = []
		for row in self.matrix:
			r.append(row[j])
		return r
		
	def rows(self):
		return len(self.matrix)

	def cols(self):
		return len(self.matrix[0])
		
	def is_square(self):
		return self.rows() == self.cols()
		
	def zeros(self, row, col):
		if not row > 0:
			raise ValueError("Invalid number of rows (given %d)" % row)
		if not col > 0:
			raise ValueError("Invalid number of columns (given %d)" % col)
		self.matrix = []
		for i in xrange(row):
			self.matrix.append([])
			for j in xrange(col):
				self.matrix[i].append(0)
				
	def determinant(self):
		assert self.is_square(), 'Can only compute the determinant of a square Matrix'
		if self.rows() == 1:
			return self.matrix[0][0]
		i = 0 # can be chosen arbitrarily (smaller than self.height)
		sum = 0
		for j in range(0, self.rows()):
			if self.matrix[i][j] == 0:
				continue
			value = (-1)**(i+j) * self.matrix[i][j] * self._A_ij(i, j).determinant()
			sum += value
		return sum
	
	def cut(self, left = 0, right = None, top = 0, bottom = None):
		if right is None:
			right = self.cols()
		if bottom is None:
			bottom = self.rows()
		assert left >= 0 and left < self.cols(), 'left out of bounds'
		assert right > 0 and right <= self.cols(), 'right out of bounds'
		assert top >= 0 and top < self.rows(), 'top out of bounds'
		assert bottom > 0 and bottom <= self.rows(), 'bottom out of bounds'
		assert left < right, 'left must be smaller than right'
		assert top < bottom, 'top must be smaller than bottom'
		width = right - left
		height = bottom - top
		flat_values = self.make_list()
		values = []
		for row in range(0, height):
			newrow = []
			for col in range(0, width):
				value = flat_values[self.cols() * top + left + self.cols() * row + col]
				newrow.append(value)
			values.append(newrow)
		return Matrix(values)	

	def _A_ij(self, i, j):
		assert i >= 0 and i < self.rows(), 'i out of bounds'
		assert j >= 0 and j < self.cols(), 'j out of bounds'
		if i == 0:
			m1 = self.cut(top=1)
		elif i == self.rows() - 1:
			m1 = self.cut(bottom=self.rows() - 1)
		else:
			tm1 = self.cut(bottom=i)
			tm2 = self.cut(top=i+1)
			m1 = stackv(tm1, tm2)
		if j == 0:
			m2 = m1.cut(left=1)
		elif j == m1.cols() - 1:
			m2 = m1.cut(right=m1.cols() - 1)
		else:
			tm1 = m1.cut(right=j)
			tm2 = m1.cut(left=j+1)
			m2 = stackh(tm1, tm2)
		return m2		

	def adjugate(self):
		"""Computes the adjugate of the Matrix"""
		assert self.is_square(), 'Can only compute the adjugate of a square Matrix'
		values = []
		for i in range(0, self.rows()):
			new_row = []
			for j in range(0, self.rows()):
				value = (-1)**(i+j) * self._A_ij(j, i).determinant()
				new_row.append(value)
			values.append(new_row)
		return Matrix(values)		

	def inverse(self):
		assert self.is_square(), 'Can only compute the inverse of a square Matrix'
		if self.rows() == 1:
			return Matrix([[operator.truediv(1,self.matrix[0][0])]])
		d = self.determinant()
		if abs(d) < 10**-4:
			raise Exception('Matrix is not invertible')
		return 1 / d * self.adjugate()
		
	def make_list(self):
		return [number for sublist in self.matrix for number in sublist]
	
	def smith_form(self):
		m = copy.deepcopy(self)
		U, G, V = Solver(m).smith_form()
		return U, G, V

	def eigenvalues(self):
		return Mat(self.matrix).eigenvals()
		
	def norm(self, type=2):
		if type == 1:
			return self._norm1()
		elif type == 2:
			return self._norm2()
		elif type == 'inf':
			return self._norm_inf()
		elif type == 'fro':
			return self._norm_fro()
		else:
			raise Exception('Illegal norm type')

	def _norm1(self):
		max = -1
		for j in range(0, self.cols()):
			value = sum(tuple(map(abs, self.col(j))))
			if value > max:
				max = value
		return max

	def _norm2(self):
		if not (self.is_row_vector() or self.is_column_vector()):
			# sqrt(dominant eigen value of A'A)
			raise FormError("Form not accepted.")
		elif self.is_row_vector():
			return math.sqrt(sum(tuple(map(lambda x: abs(x**2), self.row(0)))))
		elif self.is_column_vector():
			return math.sqrt(sum(tuple(map(lambda x: abs(x**2), self.col(0)))))

	def _norm_inf(self):
		max = -1
		for i in range(0, self.rows()):
			value = sum(tuple(map(abs, self.row(i))))
			if value > max:
				max = value
		return max

	def _norm_fro(self):
		sum = 0
		for i in range(0, self.rows()):
			for j in range(0, self.cols()):
				value = self.matrix[i][j]
				sum += abs(value**2)
		return math.sqrt(sum)
		
	def transpose(self):
		return Matrix([self.col(i) for i in range(0, self.cols())])
		
	@classmethod
	def make_random(cls, m, n, low=0, high=10):
		rows = []
		for x in range(m):
			rows.append([random.randrange(low, high) for i in range(n)])
		return Matrix(rows)

	@classmethod
	def read_console(cls):
		print 'Enter matrix row by row. Type "q" to quit.'
		rows = []
		while True:
			line = sys.stdin.readline().strip()
			if line=='q': break
			row = [int(number) for number in line.split()]
			rows.append(row)
		return Matrix(rows)

	@classmethod
	def read_file(cls, fname):
		rows = []
		for line in open(fname).readlines():
			row = [int(number) for number in line.split()]
			rows.append(row)
		return Matrix(rows)
		
	@classmethod
	def identity(cls, rank):
		matrix = Matrix(rank)
		for index in xrange(matrix.rows()):
			matrix[(index, index)] = 1
		return matrix

class NumberSystem:
	def __init__(self, matrix, digitSet):
		if not matrix.is_square():
			raise DeterminantError()
		if matrix.determinant() == 0:
			raise InverseError()
		#self.lattice = lattice
		self.matrix = matrix
		self.digitSet = digitSet

	def hash_function(self, U, G):
		s = self.find_in_diagonal(G, 1)
		return sum((U[(i,0)] % G[(i,i)]) * prod(G[(j,j)] for j in range(s, i)) for i in range(s, U.rows()))

	def is_congruent(self, elementOne, elementTwo):
		U, G, V = self.matrix.smith_form()
		return self.hash_function(U*elementOne, G) == self.hash_function(U*elementTwo, G)

	def find_congruent(self, element):
		for i in self.digitSet:
			if self.is_congruent(i, element):
				return i
		
	def phi(self, element, n = 1, save = False):
		digSet = []
		for i in range(n):
			M_inv = self.matrix.inverse()
			d = self.find_congruent(element)
			digSet.append(d)
			k = M_inv*(element-d)
			element = int(k[(0,0)])
		if save:
			return digSet
		else:
			return element
		
	def find_in_diagonal(self, G, number):
		for index in xrange(G.rows()):
			if G[(index, index)] != number:
				return index 
		return G.rows()

	def is_complete_residues_system(self):
		for i in self.digitSet:
			if any(self.is_congruent(i,j) for j in self.digitSet - {i}):
				return False
		return True

	def is_expansive(self):
		eigens = self.matrix.eigenvalues()
		return all((abs(i)>1) for i in eigens)

	def unit_condition(self):
		n = self.matrix.rows()
		tmp = self.matrix.identity(n) - self.matrix
		return abs(tmp.determinant()) != 1

	def check(self):
		if self.is_expansive() and self.is_complete_residues_system():
			if self.unit_condition():
				return True
			else:
				print "It is okay, but... unit_condition failed"
		return False

class Solver:
	def __init__(self, matrix):
		self.matrix =  matrix
		
	def leftmult2(self, m, i0, i1, a, b, c, d):
		for j in range(self.matrix.cols()):
			x, y = m[(i0,j)], m[(i1,j)]
			m[(i0,j)] = a * x + b * y
			m[(i1,j)] = c * x + d * y
	 
	def rightmult2(self, m, j0, j1, a, b, c, d):
		for i in range(self.matrix.rows()):
			x, y = m[(i,j0)], m[(i,j1)]
			m[(i,j0)] = a * x + c * y
			m[(i,j1)] = b * x + d * y
	 
	def smith_form(self, domain=ZZ):
		s = Matrix.identity(self.matrix.rows())
		t = Matrix.identity(self.matrix.cols())
		last_j = -1
		for i in range(self.matrix.rows()):
			for j in range(last_j+1, self.matrix.cols()):
				if any(i != 0 for i in self.matrix.col(j)):
					break
			else:
				break
			if self.matrix[(i,j)] == 0:
				for ii in range(self.matrix.rows()):
					if self.matrix[ii][j] != 0:
						break
				self.leftmult2(self.matrix, i, ii, 0, 1, 1, 0)
				self.rightmult2(s, i, ii, 0, 1, 1, 0)
			self.rightmult2(self.matrix, j, i, 0, 1, 1, 0)
			self.leftmult2(t, j, i, 0, 1, 1, 0)
			j = i
			upd = True
			while upd:
				upd = False
				for ii in range(i+1, self.matrix.rows()):
					if self.matrix[(ii,j)] == 0:
						continue
					upd = True
					if domain.rem(self.matrix[ii, j], self.matrix[i, j]) != 0:
						coef1, coef2, g = domain.gcdex(self.matrix[i,j], self.matrix[ii, j])
						coef3 = domain.quo(self.matrix[ii, j], g)
						coef4 = domain.quo(self.matrix[i, j], g)
						self.leftmult2(self.matrix,i, ii, coef1, coef2, -coef3, coef4)
						self.rightmult2(s, i, ii, coef4, -coef2, coef3, coef1)
					coef5 = domain.quo(self.matrix[ii, j], self.matrix[i, j])
					self.leftmult2(self.matrix, i, ii, 1, 0, -coef5, 1)
					self.rightmult2(s, i, ii, 1, 0, coef5, 1)
				for jj in range(j+1, self.matrix.cols()):
					if self.matrix[i, jj] == 0:
						continue
					upd = True
					if domain.rem(self.matrix[i, jj], self.matrix[i, j]) != 0:
						coef1, coef2, g = domain.gcdex(self.matrix[i,j], self.matrix[i, jj])
						coef3 = domain.quo(self.matrix[i, jj], g)
						coef4 = domain.quo(self.matrix[i, j], g)
						self.rightmult2(self.matrix, j, jj, coef1, -coef3, coef2, coef4)
						self.leftmult2(t, j, jj, coef4, coef3, -coef2, coef1)
					coef5 = domain.quo(self.matrix[i, jj], self.matrix[i, j])
					self.rightmult2(self.matrix, j, jj, 1, -coef5, 0, 1)
					self.leftmult2(t, j, jj, 1, coef5, 0, 1)
			last_j = j
		for i1 in range(min(self.matrix.rows(), self.matrix.cols())):
			for i0 in reversed(range(i1)):
				coef1, coef2, g = domain.gcdex(self.matrix[i0, i0], self.matrix[i1,i1])
				if g == 0:
					continue
				coef3 = domain.quo(self.matrix[i1, i1], g)
				coef4 = domain.quo(self.matrix[i0, i0], g)
				self.leftmult2(self.matrix, i0, i1, 1, coef2, coef3, coef2*coef3-1)
				self.rightmult2(s, i0, i1, 1-coef2*coef3, coef2, coef3, -1)
				self.rightmult2(self.matrix, i0, i1, coef1, 1-coef1*coef4, 1, -coef4)
				self.leftmult2(t, i0, i1, coef4, 1-coef1*coef4, 1, -coef1)
		return (s, self.matrix, t)
		
class MatrixTests(unittest.TestCase):
	def setUp(self):
		self.v1 = Matrix([[1, 2, 3]])
		self.v2 = Matrix([[4, 5, 6]])
		self.m1 = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
		self.m2 = Matrix([[4, 1, -7, 2], [-1, 9, 6, 3]])
		self.m3 = Matrix([[8, -3, 1], [4, -6, 2], [7, 3, 5], [-2, -5, 1]])
		
	def test_add_1(self):
		m1 = Matrix([[1, 2, 3], [4, 5, 6]])
		m2 = Matrix([[7, 8, 9], [10, 11, 12]])		
		m3 = m1 + m2
		self.assertTrue(m3 == Matrix([[8, 10, 12], [14,16,18]]))
		
	def test_add_2(self):
		m1 = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
		m2 = Matrix([[7, 8, 9], [10, 11, 12], [13, 14, 15]])		
		m3 = m1 + m2
		self.assertTrue(m3 == Matrix([[8, 10, 12], [14,16,18], [20, 22, 24]]))
		
	def test_add_3(self):
		m1 = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
		m2 = Matrix.identity(3)		
		m3 = m1 + m2
		self.assertTrue(m3 == Matrix([[2, 2, 3], [4, 6, 6], [7, 8, 10]]))

	def test_sub(self):
		m1 = Matrix([[1, 2, 3], [4, 5, 6]])
		m2 = Matrix([[7, 8, 9], [10, 11, 12]])		
		m3 = m2 - m1
		self.assertTrue(m3 == Matrix([[6, 6, 6], [6, 6, 6]]))

	def test_mul(self):
		m1 = Matrix([[1, 2, 3], [4, 5, 6]])
		m2 = Matrix([[7, 8], [10, 11], [12, 13]])
		id = Matrix.identity(3)
		self.assertTrue(m1 * m2 == Matrix([[63, 69], [150, 165]]))
		self.assertTrue(m2 * m1 == Matrix([[39, 54, 69], [54, 75, 96], [64, 89, 114]]))
		self.assertTrue(m1 * id == m1)
		self.assertTrue(id * m2 == m2)

	def test_det(self):
		m1 = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
		m2 = Matrix([[-1, -1], [1, -1]])
		m3 = Matrix([[6,1,1],[4,-2,5],[2,8,7]])
		id = Matrix.identity(3)
		self.assertTrue(m1.determinant() == 0)
		self.assertTrue(m2.determinant() == 2)
		self.assertTrue(m3.determinant() == -306)
		self.assertTrue(id.determinant() == 1)

	def test_inv(self):
		self.assertEqual(Matrix([[1, 3, 3], [1, 4, 3], [1, 3, 4]]).inverse(), Matrix([[7, -3, -3], [-1, 1, 0], [-1, 0, 1]]))
		self.assertEqual(Matrix([[1, 2, 3], [0, 1, 4], [5, 6, 0]]).inverse(), Matrix([[-24, 18, 5], [20, -15, -4], [-5, 4, 1]]))
		
	def test_norm(self):
		self.assertAlmostEqual(self.m3.norm(type=1), 21)
		self.assertAlmostEqual(self.v1.norm(type=2), 3.741657387)
		self.assertAlmostEqual(self.v2.norm(type=2), 8.774964387)
		#self.assertAlmostEqual(self.m3.norm(type=2), 12.51910405)
		self.assertAlmostEqual(self.m3.norm(type='inf'), 15)
		self.assertAlmostEqual(self.m3.norm(type='fro'), 15.58845727)
		self.assertEqual(self.v1.norm(), self.v1.transpose().norm())
		self.assertEqual(self.v2.norm(), self.v2.transpose().norm())
		self.assertRaises(Exception, self.v1.norm, type='non-existant')
		
		
class NumberSystemTests(unittest.TestCase):
	def test_unit_condition(self):
		mat1 = Matrix([[6,1,1],[4,-2,5],[2,8,7]])
		digitSet1 = {0,1,2,3,4,5,6,7,8,9}

		mat2 = Matrix([[-1,-1],[1,-1]])
		digitSet2 = {(0,0),(1,0)}
		
		self.assertTrue(NumberSystem(mat1, digitSet1).unit_condition() == True)
		self.assertTrue(NumberSystem(mat2, digitSet2).unit_condition() == True)
		self.assertTrue(NumberSystem(Matrix([[2,-1],[1,2]]), {(0,0),(1,0),(0,1),(0,-1)}).unit_condition() == True)
		
	def test_find_in_diagonal(self):
		mat1 = Matrix([[6,1,1],[4,-2,5],[2,8,7]])
		mat2 = Matrix([[1,1,1],[4,-2,5],[2,8,7]])
		mat3 = Matrix([[1,1,1],[4,1,5],[2,8,7]])
		mat4 = Matrix([[1,1,1],[4,1,5],[2,8,1]])
		digitSet = {0,1,2,3,4,5,6,7,8,9}
		self.assertTrue(NumberSystem(mat1, digitSet).find_in_diagonal(mat1, 1) == 0)
		self.assertTrue(NumberSystem(mat1, digitSet).find_in_diagonal(mat2, 1) == 1)
		self.assertTrue(NumberSystem(mat1, digitSet).find_in_diagonal(mat3, 1) == 2)
		self.assertTrue(NumberSystem(mat1, digitSet).find_in_diagonal(mat4, 1) == 3)

	def test_is_congruent(self):
		numsys = NumberSystem(Matrix([[10]]), {0,1,2,3,4,5,6,7,8,9})
		
		self.assertTrue(numsys.is_congruent(10,0) == True)
		self.assertTrue(numsys.is_congruent(14,4) == True)
		self.assertTrue(numsys.is_congruent(6,36) == True)
		self.assertFalse(numsys.is_congruent(13,12) == True)
		self.assertFalse(numsys.is_congruent(66,43) == True)
		
		numsys2 = NumberSystem(Matrix([[-1,-1],[1,-1]]), {(0,0),(1,0)})
		self.assertFalse(numsys2.is_congruent((0,0),(1,0)) == True)

	def test_is_complete_residues_system(self):
		self.assertTrue(NumberSystem(Matrix([[10]]), {0,1,2,3,4,5,6,7,8,9}).is_complete_residues_system() == True)
		self.assertTrue(NumberSystem(Matrix([[10]]), {0,1,2,3,4,5,6,7,8,89}).is_complete_residues_system() == True)
		self.assertFalse(NumberSystem(Matrix([[10]]), {0,1,2,3,4,5,6,7,8,28}).is_complete_residues_system() == True)
		self.assertTrue(NumberSystem(Matrix([[-1,-1],[1,-1]]), {(0,0),(1,0)}).is_complete_residues_system() == True)
	
	def test_find_congruent(self):
		numsys = NumberSystem(Matrix([[-1,-1],[1,-1]]), {(0,0),(1,0)})
		self.assertTrue(numsys.find_congruent((0,1)) == (1,0))
		self.assertTrue(numsys.find_congruent((1,1)) == (0,0))
		
		numsys2 = NumberSystem(Matrix([[10]]), {0,1,2,3,4,5,6,7,8,9})
		self.assertTrue(numsys2.find_congruent(13) == 3)
		self.assertTrue(numsys2.find_congruent(15) == 5)
		self.assertTrue(numsys2.find_congruent(64) == 4)
		self.assertTrue(numsys2.find_congruent(8486) == 6)
	
	def test_check(self):
		self.assertTrue(NumberSystem(Matrix([[10]]), {0,1,2,3,4,5,6,7,8,9}).check() == True)
		self.assertTrue(NumberSystem(Matrix([[-1,-1],[1,-1]]), {(0,0),(1,0)}).check() == True)
		
def stackh(*matrices):
	matrices = _normalize_args(matrices)
	assert len(matrices) > 0, 'Can\'t stack zero matrices'
	for matrix in matrices:
		assert isinstance(matrix, Matrix), 'Can only stack matrices'
	height = matrices[0].rows()
	for matrix in matrices:
		assert matrix.rows() == height, 'Can\'t horizontally stack matrices with different heights'
	values = []
	for row in range(0, height):
		newrow = []
		for matrix in matrices:
			newrow += matrix.row(row)
		values.append(newrow)
	return Matrix(values)

def stackv(*matrices):
	matrices = _normalize_args(matrices)
	assert len(matrices) > 0, 'Can\'t stack zero matrices'
	for matrix in matrices:
		assert isinstance(matrix, Matrix), 'Can only stack matrices'
	width = matrices[0].cols()
	for matrix in matrices:
		assert matrix.cols() == width, 'Can\'t vertically stack matrices with different widths'
	values = []
	for matrix in matrices:
		values += matrix.matrix
	return Matrix(values)		

def _normalize_args(matrices):
	if len(matrices) > 0:
		first_elem = matrices[0]
		if isinstance(first_elem, list) or isinstance(first_elem, tuple):
			assert len(matrices) == 1, 'Couldn\'t normalize arguments'
			return first_elem
		return matrices
	return matrices

test = True

if test:
	if __name__ == "__main__":
		unittest.main()
else:
	v1 = Matrix([[1, 2, 3]])
	v2 = Matrix([[4, 5, 6]])
	print v2.is_vector()
