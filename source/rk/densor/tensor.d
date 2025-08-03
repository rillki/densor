module rk.densor.tensor;

import std.array : array;
import std.format : format;

import mir.ndslice;

import rk.densor.util;

template tensor(size_t[] Shape, T = float)
{
    /// Create a new zero-initilized tensor.
    auto tensor()
    {
        return new Tensor!(T, Shape)();
    }

    /// Create new tensor with default value initialized.
    auto tensor(T value)
    {
        return new Tensor!(T, Shape)(value);
    }

    /// Create tensor from existing data.
    auto tensor(T[] data)
    {
        return new Tensor!(T, Shape)(data);
    }

    /// Create tensor from RoR data (array of arrays).
    auto tensor(RoR)(RoR data)
    {
        return new Tensor!(T, Shape)(data);
    }

    /// Create tensor from slice.
    auto tensor(Slice!(T*, Shape.length) slice)
    {
        return new Tensor!(T, Shape)(slice);
    }
}

class Tensor(T = float, size_t[] Shape)
{
    /// make aliases
    alias SliceType = Slice!(T*, Shape.length);

    /// slice values
    SliceType data;

    /// Create a new zero-initilized tensor.
    this()
    {
        this(0);
    }

    /// Create new tensor with default value initialized.
    this(T value)
    {
        auto tmp = slice!T(Shape);
        tmp[] = value;
        this.data = tmp;
    }

    /// Create tensor from existing data.
    this(T[] data)
    in(Shape.totalElements == data.length, format!"Mismatch in the number of elements shape(%s) and data(%s)!"(
        Shape, data.length))
    {
        this.data = data.sliced(Shape);
    }

    /// Create tensor from RoR data (array of arrays).
    this(RoR)(RoR data)
    {
        this(data.fuse.as!T.slice);
    }

    /// Create tensor from slice.
    this(SliceType slice) 
    in(Shape.length == slice.shape.length, format!"Cannot reshape slice! Incompatible dimensions: %s != %s"(
        Shape.length, slice.shape.length))
    {
        import std.array : array;
        this(slice.flattened.array);
    }

    /// Return shape of tensor.
    auto shape() => this.data.shape;

    /// Return number of dimensions.
    auto ndim() => this.data.shape.length;
}

unittest
{
    import std.stdio;

    auto v = new Tensor!(float, [2, 2])([1, 2, 3, 4]);
    assert(v.shape == [2, 2]);
    assert(v.ndim == 2);
    assert(v.data[] == [[1, 2], [3, 4]]);

    auto w = tensor!([2, 2])([[1, 2], [3, 4]]);
    assert(w.shape == [2, 2]);
    assert(w.ndim == 2);
    assert(w.data[] == [[1, 2], [3, 4]]);

    auto x = tensor!([4])(1);
    assert(x.shape == [4]);
    assert(x.ndim == 1);
    assert(x.data[] == [1, 1, 1, 1]);

    auto m = new Tensor!(float, [4, 1])(v.data);
    v.data[] = -1;
    assert(m.shape == [4, 1]);
    assert(m.ndim == 2);
    assert(m.data[] == [[1], [2], [3], [4]]);

    // auto k = new Tensor!(float, [1, 4])(x.data);
    // x.data[] = -1;
    // assert(k.shape == [1, 4]);
    // assert(k.ndim == 2);
    // assert(k.data[] == [[1, 2, 3, 4]]);

    
    // auto w = new Tensor!(float, [2, 2])(v);
    // v.data[] = 12;
    // writeln(v.data);

    // auto x = tensor!([2, 2])(-3);
    // auto y = tensor!([2, 2])([1, 2, 3, 4]);
    // auto z = tensor!([2, 2])(x);
    // x.data[] = 1;
    // writeln(x.data, "\n", z.data);

}

