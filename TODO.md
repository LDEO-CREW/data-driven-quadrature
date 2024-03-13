# TODOs:

- Allow for greater flexibility in cost functions
- Restructure annealing loop and data recording
- Add random seed selection for results testing
- Add required package in  <code>README.md</code>
- Provide benchmarking (checks)
    - Cost function
    - Mapping function
    - Valid Inputs
    - Many of the TODOs in ddq.py
- Build documentation and use case
- Are weights assigned per point per axis? If so, the number of weights would then be (# axes * # points) --> this would make it much harder with CVXPY
    - It would make sense this way as the weights along a single integration axis must sum to the axes length
    - Does this change with normalized weights? Then the weights of each point could sum to 1 along every axis!
- Add in point uniqueness condition (may need to restructure point set data structure)
- Why not replace in line 121 (SA_functions.py)?
- Ask about the structure of blocks in the code

# TODONEs:
- Build sandbox problem to test with (03/02)
- Remove non-single integration axis point selection (03/02)
- Remove non-linear feature mapping (03/06)
- Finish building the optimization loop (03/08)
- Get one integration axis working (03/09)
- Get CVXPY working with Paulina's reduced data (03/09)