import unittest
from field import FiniteField, Polynomial

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

if __name__ == '__main__':
    unittest.main()