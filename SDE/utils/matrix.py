import numpy as np  # type: ignore

################################################################
####################### MATRIX ENVIRONMENT #####################
################################################################

class Matrix:
    m: np.array

    ##################### CONSTRUCTORS #####################

    def __init__(self, matrix):
        """
        Initialize the matrix as a numpy array.
        """
        self.m = np.array(matrix)

    @classmethod
    def empty_matrix(cls, n, m):
        """
        Create an empty matrix with dimensions n x m.

        n (int): Number of rows.
        m (int): Number of columns.
        Returns: empty matrix with dimensions n x m.
        """

        zero_matrix = np.zeros((n, m))
        return cls(zero_matrix)

    ################### getters #####################

    def get_row(self, row_index):
        return self.m[row_index]

    def get_col(self, col_index):
        return [row[col_index] for row in self.m]

    def get_shape(self):
        """
        Returns the shape of the matrix as a tuple (rows, columns).
        """
        # Check if the matrix is not empty
        if self.m.size > 0:
            rows = len(self.m)
            columns = len(self.m[0])
        else:
            rows, columns = 0, 0  # If matrix is empty
        return (rows, columns)

    ################ operator overloading ################

    def __getitem__(self, indices):
        # surcharing operator [,]
        if isinstance(indices, tuple) and len(indices) == 2:
            i, j = indices
            return self.m[i][j]
        else:
            raise IndexError("wrong format -> use [.,.]")

    def __setitem__(self, pos, value):
        i, j = pos
        self.m[i][j] = value

    def __str__(self):
        """
        str representation for the matrix class
        """
        rows = []
        for row in self.m:
            rows.append(" ".join(str(elem) for elem in row))
        return "\n".join(rows)

    #################### METHODS ####################

    def compute_determinant(self, matrix=None):
        """
        Computes the determinant of the matrix.
        """
        # Initial matrix
        if matrix is None:
            matrix = self.m
        # Get matrix size
        rows, columns = self.get_shape()
        # Handle errors
        if rows == 0 or columns == 0:
            return 0  # Empty matrix
        if rows != columns:
            raise ValueError("Determinant can only be computed for square matrices.")  # Non square matrix
        # Compute determinant of matrix size = 1
        if rows == 1 and columns == 1:
            return matrix[0, 0]
        # Compute determinant of matrix size = 2
        if rows == 2 and columns == 2:
            return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
        # Compute determinant of matrix size > 2
        determinant = 0
        for c in range(columns):
            # Get minor matrix by excluding row 1 and column c
            minor_matrix = Matrix(np.concatenate((self.m[1:, :c], self.m[1:, c + 1:]), axis=1))
            # Compute the determinant of the minor
            determinant += ((-1) ** c) * self.m[0, c] * minor_matrix.compute_determinant()
        return determinant

    def is_invertible(self):
        """
        Determines if the matrix is invertible.
        """
        return self.compute_determinant() != 0

    def cholesky_decomposition(self):
        """
        Computes the Cholesky decomposition of the matrix.
        """
        # Get matrix size
        rows, columns = self.get_shape()
        # Initialize L matrix
        L = np.zeros((rows, rows))
        for i in range(rows):
            for j in range(i + 1):
                # Calculate the sum of squares of elements already calculated in L
                sum_k = np.dot(L[i, :j], L[j, :j])
                if i == j:
                    # Diagonal elements are the square root of the difference
                    L[i, j] = np.sqrt(max(self.m[i, i] - sum_k, 0))
                else:
                    # Non-diagonal elements calculated by division
                    L[i, j] = (1.0 / L[j, j] * (self.m[i, j] - sum_k)) if L[j, j] > 0 else 0
        return L

    def qr_decomposition(self):
        """
        Computes the orthogonal (QR) decomposition of the matrix.
        """
        # Get matrix size
        rows, columns = self.get_shape()
        # Initialize matrices
        Q = np.zeros((rows, columns), dtype='float64')
        R = np.zeros((columns, columns), dtype='float64')
        for j in range(columns):
            v = np.array(self.m[:, j], dtype='float64')  # Copy of column j
            for i in range(j):
                # Project v onto column i of Q and subtract
                R[i, j] = np.dot(Q[:, i], self.m[:, j])
                v -= R[i, j] * Q[:, i]
            R[j, j] = np.linalg.norm(v)  # Diagonal elements of R are the norms of the vectors
            Q[:, j] = v / R[j, j]  # Normalize v to get the orthogonal vectors
        return Q

    def orthogonal_decomposition(self):
        def sqrt_diagonal_matrix(D):
            # Compute square root of each diagonal element
            sqrt_D = np.sqrt(np.diag(D))
            # Construct the square root diagonal matrix
            sqrt_D_matrix = np.diag(sqrt_D)
            return sqrt_D_matrix

        # Eigen decomposition of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(self.m)

        # Construct diagonal matrix D
        D = np.diag(eigenvalues)

        # Construct orthogonal matrix O
        O = eigenvectors

        B = np.dot(O, sqrt_diagonal_matrix(D))
        return B
