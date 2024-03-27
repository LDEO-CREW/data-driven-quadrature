# TODOs:

- Add random seed selection for results testing
- Add required package in  <code>README.md</code>
- Provide benchmarking (checks)
    - Cost function
    - Mapping function
    - Valid Inputs
    - Many of the TODOs in ddq.py
- Build documentation and use case
-
# TODONEs:
- Build sandbox problem to use for tests                                                    (03/02)
- Remove non-single integration axis point selection                                        (03/02)
- Remove non-linear feature mapping                                                         (03/06)
- Finish building the optimization loop                                                     (03/08)
- Get one integration axis working                                                          (03/09)
- Get CVXPY working with Paulina's reduced data                                             (03/09)
- Graph average cost per block                                                              (03/14)
- Restructure annealing loop and data recording                                             (03/14)
- Add in point uniqueness condition (may need to restructure point set data structure)      (03/17)
- Allow for greater flexibility in cost functions                                           (03/20)
- Cost function benchmarking                                                                (03/21)
- Parameter benchmarking                                                                    (03/21)
- Generalized error message outputs                                                         (03/21)
- Cleaned up optimize function and finished respective TODOs                                (03/23)

# MISC Notes:
- Points are returned in order of the given integration axes
- Vector from mapping function must be flattened (Therefore y_ref must be a flat vector) AND normalized to integration axes

- How do we handle multi-variate data and cost functions?
