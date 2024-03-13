# data-driven-quadrature

### Variable Definitions
<ul>
    <li> <code>x</code>: Input data to directly choose integration points from.
    <li> <code>x_sup</code>: Any supplementary data necessary for mapping function. Not required.
    <li> <code>y_ref</code>: Reference data for cost calculation after mapping and integration.
    <li> <code>C</code>: Cost function. Must take two objects of shape <code>y_ref</code> and return a scalar cost. Cost function should be minimized (ideally 0) for two identical objects.
    <li> <code>M</code>: Mapping function that takes <code>x</code>, <code>x_sup</code>, and a set of integration points <code>p</code> and returns a vector (list) <code>v</code> of transformed points to 
    <li> <code>params</code>: Additional parameters in a dictionary including:
    <ul>
        <li> <code>n_points</code>: Number of integration points to select, each following <code>integration_dict</code>
        <li> <code>integration_dict</code>: Dictionary with <b>keys</b> being axes labels to select <b>value</b> number of values along each integration axis. 
    </ul>
</ul>


### Function Definition