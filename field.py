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
        a = torch.nn.functional.pad(
            self.coefficients, (0, max_len - len(self.coefficients)), value=0
        )
        b = torch.nn.functional.pad(
            other.coefficients, (0, max_len - len(other.coefficients)), value=0
        )
        result = (a + b) % self.field.prime
        return Polynomial(result.tolist(), self.field)

    def multiply(self, other):
        # FFT-based multiplication
        n = len(self.coefficients) + len(other.coefficients) - 1
        n_fft = 1 << (n - 1).bit_length()  # Next power of 2
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


class MultivariatePolynomial:
    def __init__(self, coefficients, variables, field: FiniteField):
        """
        coefficients: dict mapping tuples of exponents to coefficients
        variables: list of variable names
        """
        self.coefficients = {tuple(k): v % field.prime for k, v in coefficients.items()}
        self.variables = variables
        self.field = field

    def add(self, other):
        result = self.coefficients.copy()
        for exponents, coeff in other.coefficients.items():
            if exponents in result:
                result[exponents] = (result[exponents] + coeff) % self.field.prime
                if result[exponents] == 0:
                    del result[exponents]
            else:
                result[exponents] = coeff
        return MultivariatePolynomial(result, self.variables, self.field)

    def multiply(self, other):
        # FFT-based multiplication with correct tensor sizing
        import torch
        import math

        # Determine the maximum exponents for each variable in self and other
        max_self_exponents = [
            max((k[i] for k in self.coefficients.keys()), default=0)
            for i in range(len(self.variables))
        ]
        max_other_exponents = [
            max((k[i] for k in other.coefficients.keys()), default=0)
            for i in range(len(self.variables))
        ]

        # Compute the size for each dimension: (max_self + max_other + 1)
        result_size = [
            s + o + 1 for s, o in zip(max_self_exponents, max_other_exponents)
        ]

        # Optionally, pad each dimension to the next power of two for FFT efficiency
        padded_size = [1 << (size - 1).bit_length() for size in result_size]

        # Initialize tensors with the padded sizes
        self_tensor = torch.zeros(padded_size, dtype=torch.float)
        other_tensor = torch.zeros(padded_size, dtype=torch.float)

        # Populate the tensors with coefficients
        for exponents, coeff in self.coefficients.items():
            if all(
                0 <= exponents[i] < padded_size[i] for i in range(len(self.variables))
            ):
                self_tensor[exponents] = coeff
        for exponents, coeff in other.coefficients.items():
            if all(
                0 <= exponents[i] < padded_size[i] for i in range(len(self.variables))
            ):
                other_tensor[exponents] = coeff

        # Perform FFT on each dimension
        self_fft = torch.fft.fftn(self_tensor)
        other_fft = torch.fft.fftn(other_tensor)

        # Element-wise multiplication in the frequency domain
        result_fft = self_fft * other_fft

        # Inverse FFT to get the convolution result
        result_ifft = torch.fft.ifftn(result_fft).real.round().long() % self.field.prime

        # Extract non-zero coefficients
        result_coefficients = {}
        it = torch.nonzero(result_ifft)
        for idx in it:
            exponents = tuple(idx.tolist())
            coeff = result_ifft[exponents].item()
            if coeff != 0:
                result_coefficients[exponents] = coeff

        return MultivariatePolynomial(result_coefficients, self.variables, self.field)

    def evaluate(self, assignments):
        """
        assignments: dict mapping variable names to values
        """
        result = 0
        for exponents, coeff in self.coefficients.items():
            term = coeff
            for var, exp in zip(self.variables, exponents):
                term = (
                    term * pow(assignments[var], exp, self.field.prime)
                ) % self.field.prime
            result = (result + term) % self.field.prime
        return result

    def __repr__(self):
        terms = []
        for exponents, coeff in sorted(self.coefficients.items()):
            term = str(coeff)
            for var, exp in zip(self.variables, exponents):
                if exp != 0:
                    term += f"*{var}^{exp}"
            terms.append(term)
        return " + ".join(terms) if terms else "0"
