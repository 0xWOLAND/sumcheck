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

    def degree(self):
        return len(self.coefficients) - 1

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

    def subtract(self, other):
        max_len = max(len(self.coefficients), len(other.coefficients))
        a = torch.nn.functional.pad(
            self.coefficients, (0, max_len - len(self.coefficients)), value=0
        )
        b = torch.nn.functional.pad(
            other.coefficients, (0, max_len - len(other.coefficients)), value=0
        )
        result = (a - b) % self.field.prime
        return Polynomial(result.tolist(), self.field)


class MultivariatePolynomial:
    def __init__(self, coefficients, variables, field: FiniteField):
        """
        coefficients: dict mapping tuples of exponents to coefficients
        variables: list of variable names
        """
        self.coefficients = {tuple(k): v % field.prime for k, v in coefficients.items()}
        self.variables = variables
        self.field = field

    def degree(self, variable=None):
        if variable:
            if variable not in self.variables:
                raise ValueError(f"Variable {variable} not found in polynomial")
            var_index = self.variables.index(variable)
            return max(
                (exponents[var_index] for exponents in self.coefficients.keys()),
                default=0,
            )
        else:
            return max(
                (sum(exponents) for exponents in self.coefficients.keys()), default=0
            )

    def add(self, other):
        if self.variables != other.variables:
            raise ValueError(
                "Cannot add polynomials with different variables or ordering."
            )
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
        if self.variables != other.variables:
            raise ValueError(
                "Cannot multiply polynomials with different variables or ordering."
            )

        result = {}
        for exp1, coeff1 in self.coefficients.items():
            for exp2, coeff2 in other.coefficients.items():
                new_exp = tuple(e1 + e2 for e1, e2 in zip(exp1, exp2))
                new_coeff = (coeff1 * coeff2) % self.field.prime
                if new_exp in result:
                    result[new_exp] = (result[new_exp] + new_coeff) % self.field.prime
                else:
                    result[new_exp] = new_coeff
                if result[new_exp] == 0:
                    del result[new_exp]
        return MultivariatePolynomial(result, self.variables, self.field)

    def evaluate(self, assignments):
        """
        assignments: dict mapping variable names to values
        """
        result = 0
        for exponents, coeff in self.coefficients.items():
            term = coeff
            for var, exp in zip(self.variables, exponents):
                if var not in assignments:
                    raise KeyError(f"Variable '{var}' not provided in assignments.")
                term = (
                    term * pow(assignments[var], exp, self.field.prime)
                ) % self.field.prime
            result = (result + term) % self.field.prime
        return result

    def partial_evaluate(self, assignments):
        """
        Partially evaluates the polynomial by substituting some variables with given values.

        :param assignments: dict mapping variable names to values
        :return: A new MultivariatePolynomial with the assigned variables evaluated.
        """
        # {{ edit_1 }} Add validation for nonexistent variables
        unknown_vars = set(assignments.keys()) - set(self.variables)
        if unknown_vars:
            raise ValueError(f"Variables {unknown_vars} not found in the polynomial.")

        remaining_vars = [var for var in self.variables if var not in assignments]
        new_coefficients = {}

        for exponents, coeff in self.coefficients.items():
            new_coeff = coeff
            new_exponents = []
            for var, exp in zip(self.variables, exponents):
                if var in assignments:
                    substitution = pow(assignments[var], exp, self.field.prime)
                    new_coeff = (new_coeff * substitution) % self.field.prime
                else:
                    new_exponents.append(exp)
            if new_coeff != 0:
                new_exponents = tuple(new_exponents)
                # If all remaining exponents are 0, represent as a constant term with empty tuple
                if all(e == 0 for e in new_exponents):
                    new_exponents = ()
                if new_exponents in new_coefficients:
                    new_coefficients[new_exponents] = (
                        new_coefficients[new_exponents] + new_coeff
                    ) % self.field.prime
                    if new_coefficients[new_exponents] == 0:
                        del new_coefficients[new_exponents]
                else:
                    new_coefficients[new_exponents] = new_coeff

        return MultivariatePolynomial(new_coefficients, remaining_vars, self.field)

    def __repr__(self):
        if not self.coefficients:
            return "0"
        terms = []
        for exponents, coeff in sorted(self.coefficients.items()):
            if coeff == 0:
                continue
            term = []
            if coeff != 1 or all(exp == 0 for exp in exponents):
                term.append(str(coeff))
            for var, exp in zip(self.variables, exponents):
                if exp == 1:
                    term.append(var)
                elif exp > 1:
                    term.append(f"{var}^{exp}")
            terms.append("*".join(term))
        return " + ".join(terms) if terms else "0"

    def subtract(self, other):
        if self.variables != other.variables:
            raise ValueError(
                "Cannot subtract polynomials with different variables or ordering."
            )
        result = self.coefficients.copy()
        for exponents, coeff in other.coefficients.items():
            if exponents in result:
                result[exponents] = (result[exponents] - coeff) % self.field.prime
                if result[exponents] == 0:
                    del result[exponents]
            else:
                result[exponents] = (-coeff) % self.field.prime
        return MultivariatePolynomial(result, self.variables, self.field)

    # {{ edit_2 }} *(Optional)* Implement equality for better testing
    def __eq__(self, other):
        if not isinstance(other, MultivariatePolynomial):
            return False
        if self.variables != other.variables:
            return False
        # Compare coefficients, ignoring zero coefficients
        self_coeffs = {k: v for k, v in self.coefficients.items() if v != 0}
        other_coeffs = {k: v for k, v in other.coefficients.items() if v != 0}
        return self_coeffs == other_coeffs
