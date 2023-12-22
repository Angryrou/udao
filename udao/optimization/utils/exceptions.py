class NoSolutionError(ValueError):
    "Raised when no solution is found for an MOO problem"
    ...


class UncompliantSolutionError(ValueError):
    "Raised when the candidate is not compliant with the optimization algorithm"
    ...
