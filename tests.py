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
        self.assertEqual(self.poly1.evaluate(2), (1 + 2 * 2 + 3 * 4) % 17)

    def test_subtract(self):
        # Test subtraction of two polynomials
        poly3 = Polynomial([1, 1, 1], self.field)  # Represents 1 + x + x^2
        result = self.poly1.subtract(poly3)
        expected = [(1 - 1) % 17, (2 - 1) % 17, (3 - 1) % 17]  # [0, 1, 2]
        self.assertEqual(result.coefficients.tolist(), expected)

    def test_degree(self):
        # Test the degree of a polynomial
        poly = Polynomial([1, 2, 3], self.field)  # Represents 1 + 2x + 3x^2
        self.assertEqual(poly.degree(), 2)


class TestMultivariatePolynomial(unittest.TestCase):
    def setUp(self):
        self.field = FiniteField(17)
        # Example: 3*x^2*y + 2*y^2
        self.poly1 = MultivariatePolynomial(
            coefficients={(2, 1): 3, (0, 2): 2}, variables=["x", "y"], field=self.field
        )
        # Example: x + 4*y
        self.poly2 = MultivariatePolynomial(
            coefficients={(1, 0): 1, (0, 1): 4}, variables=["x", "y"], field=self.field
        )

    def test_add(self):
        result = self.poly1.add(self.poly2)
        expected = {(2, 1): 3, (0, 2): 2, (1, 0): 1, (0, 1): 4}
        self.assertEqual(result.coefficients, expected)

    def test_multiply_fft(self):
        # Multiply using FFT-based method
        result = self.poly1.multiply(self.poly2)
        expected = {
            (3, 1): 3,  # (2,1) + (1,0) => x^3*y
            (2, 2): 12,  # (2,1) + (0,1) => x^2*y^2
            (1, 2): 2,  # (1,0) + (0,2) => x*y^2
            (0, 3): 8,  # (0,2) + (0,1) => y^3
        }
        # Apply modulo 17
        expected_mod = {k: v % 17 for k, v in expected.items()}
        self.assertEqual(result.coefficients, expected_mod)

    def test_evaluate(self):
        assignments = {"x": 2, "y": 3}
        # 3*(2)^2*(3) + 2*(3)^2 = 3*4*3 + 2*9 = 36 + 18 = 54 % 17 = 54 - 3*17 = 54 - 51 = 3
        self.assertEqual(self.poly1.evaluate(assignments), 3)

    def test_subtract(self):
        # Test subtraction of two multivariate polynomials
        poly3 = MultivariatePolynomial(
            coefficients={(2, 1): 1, (0, 1): 4}, variables=["x", "y"], field=self.field
        )
        result = self.poly1.subtract(poly3)
        expected = {
            (2, 1): (3 - 1) % 17,  # 3 - 1 = 2
            (0, 2): 2,  # 2 - 0 = 2
            (0, 1): (-4) % 17,  # 0 - 4 = -4 mod 17 = 13
        }
        self.assertEqual(result.coefficients, expected)

    def test_zero_polynomial_addition(self):
        # Test adding a zero polynomial
        zero_poly = MultivariatePolynomial({}, ["x", "y"], self.field)
        result = self.poly1.add(zero_poly)
        self.assertEqual(result.coefficients, self.poly1.coefficients)

    def test_zero_polynomial_multiplication(self):
        # Test multiplying by a zero polynomial
        zero_poly = MultivariatePolynomial({}, ["x", "y"], self.field)
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
            coefficients={(1, 1): 5}, variables=["x", "y"], field=self.field
        )
        sum1 = self.poly1.add(self.poly2).add(poly3)
        sum2 = self.poly1.add(self.poly2.add(poly3))
        self.assertEqual(sum1.coefficients, sum2.coefficients)

    def test_evaluate_with_missing_variable(self):
        # Test evaluation with missing variables in assignments
        assignments = {"x": 2}  # Missing 'y'
        with self.assertRaises(KeyError):
            self.poly1.evaluate(assignments)

    def test_evaluate_zero_polynomial(self):
        # Test evaluating a zero polynomial
        zero_poly = MultivariatePolynomial({}, ["x", "y"], self.field)
        result = zero_poly.evaluate({"x": 1, "y": 1})
        self.assertEqual(result, 0)

    def test_multiply_multiple_terms_fft(self):
        # Multiply polynomials with multiple terms using FFT-based method
        poly3 = MultivariatePolynomial(
            coefficients={(1, 1): 5, (2, 2): 7}, variables=["x", "y"], field=self.field
        )
        result = self.poly1.multiply(poly3)
        expected = {
            (3, 2): (3 * 5) % 17,  # (2,1) + (1,1) => x^3*y^2
            (4, 3): (3 * 7) % 17,  # (2,1) + (2,2) => x^4*y^3
            (1, 3): (2 * 5) % 17,  # (0,2) * (1,1) = (1,3)
            (2, 4): (2 * 7) % 17,  # (0,2) * (2,2) = (2,4)
        }
        expected_mod = {k: v % 17 for k, v in expected.items()}
        self.assertEqual(result.coefficients, expected_mod)

    def test_degree_x(self):
        # Test the degree of the multivariate polynomial with respect to 'x'
        poly = MultivariatePolynomial(
            coefficients={(2, 1): 3, (0, 2): 2},  # Represents 3x^2y + 2y^2
            variables=["x", "y"],
            field=self.field,
        )
        self.assertEqual(poly.degree("x"), 2)

    def test_degree_y(self):
        # Test the degree of the multivariate polynomial with respect to 'y'
        poly = MultivariatePolynomial(
            coefficients={(2, 1): 3, (0, 2): 2},  # Represents 3x^2y + 2y^2
            variables=["x", "y"],
            field=self.field,
        )
        self.assertEqual(poly.degree("y"), 2)

    def test_partial_evaluate(self):
        field = FiniteField(17)
        # Polynomial: 3*x^2*y + 2*y^2
        poly = MultivariatePolynomial(
            coefficients={(2, 1): 3, (0, 2): 2}, variables=["x", "y"], field=field
        )
        # Partial evaluate y = 3
        partial = poly.partial_evaluate({"y": 3})
        # Expected: 3*x^2*3 + 2*3^2 = 9*x^2 + 18 mod 17 => 9*x^2 + 1
        expected = MultivariatePolynomial(
            coefficients={(2,): 9, (): 1}, variables=["x"], field=field
        )
        self.assertEqual(partial.coefficients, expected.coefficients)
        self.assertEqual(partial.variables, expected.variables)

    def test_partial_evaluate_all_variables(self):
        field = FiniteField(17)
        # Polynomial: x + y
        poly = MultivariatePolynomial(
            coefficients={(1, 0): 1, (0, 1): 1}, variables=["x", "y"], field=field
        )
        # Partial evaluate x = 5, y = 12
        partial = poly.partial_evaluate({"x": 5, "y": 12})
        # Expected: 5 + 12 = 17 mod 17 = 0
        expected = MultivariatePolynomial(
            coefficients={(): 0}, variables=[], field=field
        )
        self.assertEqual(partial.coefficients, {})
        self.assertEqual(partial.variables, expected.variables)

    def test_partial_evaluate_no_assignments(self):
        field = FiniteField(17)
        # Polynomial: x*y
        poly = MultivariatePolynomial(
            coefficients={(1, 1): 1}, variables=["x", "y"], field=field
        )
        # Partial evaluate with no assignments
        partial = poly.partial_evaluate({})
        # Expected: same as original
        expected = poly
        self.assertEqual(partial.coefficients, expected.coefficients)
        self.assertEqual(partial.variables, expected.variables)

    def test_partial_evaluate_some_zero_coefficients(self):
        field = FiniteField(17)
        # Polynomial: x^2 + y^2 + z^2
        poly = MultivariatePolynomial(
            coefficients={(2, 0, 0): 1, (0, 2, 0): 1, (0, 0, 2): 1},
            variables=["x", "y", "z"],
            field=field,
        )
        # Partial evaluate x = 0
        partial = poly.partial_evaluate({"x": 0})
        # Expected: y^2 + z^2
        expected = MultivariatePolynomial(
            coefficients={(2, 0): 1, (0, 2): 1}, variables=["y", "z"], field=field
        )
        self.assertEqual(partial.coefficients, expected.coefficients)
        self.assertEqual(partial.variables, expected.variables)


if __name__ == "__main__":
    unittest.main()
