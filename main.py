import random
from field import FiniteField, MultivariatePolynomial, Polynomial


field = FiniteField(17)


class Prover:
    def __init__(self, g: MultivariatePolynomial):
        self.g = g
        self.c_1 = g.evaluate_on_boolean_hypercube()
        self.num_variables = g.arity()
        self.r = []

    def step(self, r_prev: int, j: int) -> Polynomial:
        if j != 0:
            self.r.append(r_prev)
            self.g = self.g.partial_evaluate()

    def compute_gj(self, j: int) -> Polynomial:
        """
        Compute g_j(X_j) = ∑(x_{j+1},...,x_v)∈{0,1}^(v-j) g(r_1,...,r_{j-1},X_j,x_{j+1},...,x_v)

        Args:
        j (int): The current step in the sum-check protocol

        Returns:
        Polynomial: The univariate polynomial g_j(X_j)
        """
        # Create a partial assignment with known r values
        partial_assignment = {
            var: r for var, r in zip(self.g.variables[: j - 1], self.r)
        }

        # Add X_j as a symbolic variable
        partial_assignment[self.g.variables[j - 1]] = "X_j"

        # Partially evaluate g with the known assignments
        g_partial = self.g.partial_evaluate(partial_assignment)

        # Sum over all possible boolean assignments for the remaining variables
        result = MultivariatePolynomial({}, [self.g.variables[j - 1]], self.g.field)
        for assignment in range(2 ** (self.num_variables - j)):
            boolean_assignment = {}
            for k, var in enumerate(self.g.variables[j:]):
                boolean_assignment[var] = (assignment >> k) & 1

            term = g_partial.partial_evaluate(boolean_assignment)
            result = result.add(term)

        # Convert the result to a univariate polynomial and return
        return result.to_single_variate()


class Verifier:
    def __init__(self, g: MultivariatePolynomial, n: int):
        self.n = n
        self.g = g
        self.c_1 = 0
        self.g_part = []
        self.r = []

    def set_c1(self, c_1):
        self.c_1 = c_1

    def verifier_step(self, g_j: Polynomial):
        r_j = random.randint(0, field.prime - 1)

        if len(self.r) == 0:
            evaluation = g_j.evaluate(0) + g_j.evaluate(1)
            if self.c_1 != evaluation:
                return False
            else:
                self.g_part.append(g_j)
                self.r.append(r_j)

                return True
        elif len(self.r) == self.n - 1:
            self.r.append(r_j)


class SumCheck:
    def __init__(self, polynomial: MultivariatePolynomial, H: FiniteField):
        self.polynomial = polynomial
        self.H = H
        self.degree = polynomial.degree("x")

    def prover_step(self, partial_assignment):
        # Compute the sum over the next variable
        # Placeholder for prover's computation
        # Implement the prover's logic here
        pass

    def verifier_step(self, responses):
        # Verify the responses from the prover
        # Placeholder for verifier's computation
        # Implement the verifier's logic here
        pass


def main():
    # x + y
    multivariate_polynomial = MultivariatePolynomial(
        {(3, 0, 0): 2, (1, 0, 1): 1, (0, 1, 1): 1}, ["x", "y", "z"], field
    )

    prover = Prover(multivariate_polynomial)
    verifier = Verifier(multivariate_polynomial, 3)

    verifier.set_c1(prover.c_1)


if __name__ == "__main__":
    main()
