# TODOs:

- Add random seed selection for results testing
- Add required package in  <code>README.md</code>
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
- Provide benchmarking (checks) and finished most of the TODOs                              (04/01)
- Add random seed selection                                                                 (04/03)
- Add solver choice                                                                         (04/03)

# MISC Notes:
- Points are returned in order of the given integration axes
- Vector from mapping function must be flattened (Therefore y_ref must be a flat vector) AND normalized to integration axes
- Data (y_ref and v returned by mapping function) must be normalized

- How do we handle multi-variate data and cost functions?

- Use example: Normalized cost of 3e-3 -> approx. 10 for unnormalized axes
    - Found after ~10 epochs. Costs converge at about 20 epochs (block size of 100, success threshold of 50)


Next steps:
- Diataxis documentation: https://diataxis.fr/
- Publish on "Read the docs"
- Implement constraints on the points in point_set