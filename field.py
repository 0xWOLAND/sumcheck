import torch

class FiniteField:
    def __init__(self, prime):
        self.prime = prime

    def add(self, a, b):
        return (a + b) % self.prime

    def subtract(self, a, b):
        return (a - b) % self.prime

    def multiply(self, a, b):
        return (a * b) % self.prime

    def inverse(self, a):
        return pow(a, -1, self.prime)

    def divide(self, a, b):
        return self.multiply(a, self.inverse(b))

class Polynomial:
    def __init__(self, coefficients, field: FiniteField):
        self.coefficients = torch.tensor(coefficients, dtype=torch.long)
        self.field = field

    def add(self, other):
        max_len = max(len(self.coefficients), len(other.coefficients))
        a = torch.nn.functional.pad(self.coefficients, (0, max_len - len(self.coefficients)), value=0)
        b = torch.nn.functional.pad(other.coefficients, (0, max_len - len(other.coefficients)), value=0)
        result = (a + b) % self.field.prime
        return Polynomial(result.tolist(), self.field)

    def multiply(self, other):
        # FFT-based multiplication
        n = len(self.coefficients) + len(other.coefficients) - 1
        n_fft = 1 << (n-1).bit_length()  # Next power of 2
        a = torch.fft.fft(self.coefficients.float(), n=n_fft)
        b = torch.fft.fft(other.coefficients.float(), n=n_fft)
        result = torch.fft.ifft(a * b).real.round().long()[:n] % self.field.prime
        return Polynomial(result.tolist(), self.field)

    def evaluate(self, x):
        result = 0
        power = 1
        for coeff in self.coefficients:
            result = (result + coeff * power) % self.field.prime
            power = (power * x) % self.field.prime
        return result
