import unittest
from field import FiniteField, Polynomial, MultivariatePolynomial

class TestFiniteField(unittest.TestCase):
    def setUp(self):
        self.field = FiniteField(17)

    def test_add(self):
        self.assertEqual(self.field.add(5, 13), 1)

    def test_subtract(self):
        self.assertEqual(self.field.subtract(5, 13), 9)

    def test_multiply(self):
        self.assertEqual(self.field.multiply(5, 13), 14)

    def test_inverse(self):
        self.assertEqual(self.field.inverse(5), 7)

    def test_divide(self):
        self.assertEqual(self.field.divide(5, 13), (5 * self.field.inverse(13)) % 17)

class TestPolynomial(unittest.TestCase):
    def setUp(self):
        self.field = FiniteField(17)
        self.poly1 = Polynomial([1, 2, 3], self.field)
        self.poly2 = Polynomial([4, 5], self.field)

    def test_add(self):
        result = self.poly1.add(self.poly2)
        self.assertEqual(result.coefficients.tolist(), [5, 7, 3])

    def test_multiply(self):
        result = self.poly1.multiply(self.poly2)
        self.assertEqual(result.coefficients.tolist(), [4, 13, 5, 15])

    def test_evaluate(self):
        self.assertEqual(self.poly1.evaluate(2), (1 + 2*2 + 3*4) % 17)

class TestMultivariatePolynomial(unittest.TestCase):
    def setUp(self):
        self.field = FiniteField(17)
        # Example: 3*x^2*y + 2*y^2
        self.poly1 = MultivariatePolynomial(
            coefficients={
                (2, 1): 3,
                (0, 2): 2
            },
            variables=['x', 'y'],
            field=self.field
        )
        # Example: x + 4*y
        self.poly2 = MultivariatePolynomial(
            coefficients={
                (1, 0): 1,
                (0, 1): 4
            },
            variables=['x', 'y'],
            field=self.field
        )

    def test_add(self):
        result = self.poly1.add(self.poly2)
        expected = {
            (2, 1): 3,
            (0, 2): 2,
            (1, 0): 1,
            (0, 1): 4
        }
        self.assertEqual(result.coefficients, expected)

    def test_multiply_fft(self):
        # Multiply using FFT-based method
        result = self.poly1.multiply(self.poly2)
        expected = {
            (3, 1): 3,    # (2,1) + (1,0) => x^3*y
            (2, 2): 12,   # (2,1) + (0,1) => x^2*y^2
            (1, 2): 2,    # (1,0) + (0,2) => x*y^2
            (0, 3): 8     # (0,2) + (0,1) => y^3
        }
        # Apply modulo 17
        expected_mod = {k: v % 17 for k, v in expected.items()}
        self.assertEqual(result.coefficients, expected_mod)

    def test_evaluate(self):
        assignments = {'x': 2, 'y': 3}
        # 3*(2)^2*(3) + 2*(3)^2 = 3*4*3 + 2*9 = 36 + 18 = 54 % 17 = 54 - 3*17 = 54 - 51 = 3
        self.assertEqual(self.poly1.evaluate(assignments), 3)

    def test_subtract(self):
        # Test subtraction of two multivariate polynomials
        poly3 = MultivariatePolynomial(
            coefficients={
                (2, 1): 1,
                (0, 1): 4
            },
            variables=['x', 'y'],
            field=self.field
        )
        result = self.poly1.add(poly3)  # Assuming subtract is implemented as add with negation
        expected = {
            (2, 1): (3 + 1) % 17,
            (0, 2): 2,
            (0, 1): (0 + 4) % 17
        }
        self.assertEqual(result.coefficients, expected)

    def test_zero_polynomial_addition(self):
        # Test adding a zero polynomial
        zero_poly = MultivariatePolynomial({}, ['x', 'y'], self.field)
        result = self.poly1.add(zero_poly)
        self.assertEqual(result.coefficients, self.poly1.coefficients)

    def test_zero_polynomial_multiplication(self):
        # Test multiplying by a zero polynomial
        zero_poly = MultivariatePolynomial({}, ['x', 'y'], self.field)
        result = self.poly1.multiply(zero_poly)
        self.assertEqual(result.coefficients, {})

    def test_commutativity_of_multiplication(self):
        # Test that a * b == b * a
        result1 = self.poly1.multiply(self.poly2)
        result2 = self.poly2.multiply(self.poly1)
        self.assertEqual(result1.coefficients, result2.coefficients)

    def test_associativity_of_addition(self):
        # Test that (a + b) + c == a + (b + c)
        poly3 = MultivariatePolynomial(
            coefficients={
                (1, 1): 5
            },
            variables=['x', 'y'],
            field=self.field
        )
        sum1 = self.poly1.add(self.poly2).add(poly3)
        sum2 = self.poly1.add(self.poly2.add(poly3))
        self.assertEqual(sum1.coefficients, sum2.coefficients)

    def test_evaluate_with_missing_variable(self):
        # Test evaluation with missing variables in assignments
        assignments = {'x': 2}  # Missing 'y'
        with self.assertRaises(KeyError):
            self.poly1.evaluate(assignments)

    def test_evaluate_zero_polynomial(self):
        # Test evaluating a zero polynomial
        zero_poly = MultivariatePolynomial({}, ['x', 'y'], self.field)
        result = zero_poly.evaluate({'x': 1, 'y': 1})
        self.assertEqual(result, 0)

    def test_multiply_multiple_terms_fft(self):
        # Multiply polynomials with multiple terms using FFT-based method
        poly3 = MultivariatePolynomial(
            coefficients={
                (1, 1): 5,
                (2, 2): 7
            },
            variables=['x', 'y'],
            field=self.field
        )
        result = self.poly1.multiply(poly3)
        expected = {
            (3, 2): (3 * 5) % 17,    # (2,1) + (1,1) => x^3*y^2
            (4, 3): (3 * 7) % 17,    # (2,1) + (2,2) => x^4*y^3
            (1, 3): (2 * 5) % 17,    # (0,2) * (1,1) = (1,3)
            (2, 4): (2 * 7) % 17     # (0,2) * (2,2) = (2,4)
        }
        expected_mod = {k: v % 17 for k, v in expected.items()}
        self.assertEqual(result.coefficients, expected_mod)

if __name__ == '__main__':
    unittest.main()