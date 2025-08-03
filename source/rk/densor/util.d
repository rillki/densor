module rk.densor.util;

import std.algorithm : reduce;

/// Return total number of elements from shape.
size_t totalElements(size_t[] shape)
{
    size_t s = 1;
    foreach (x; shape) s *= x;
    return s;
}
