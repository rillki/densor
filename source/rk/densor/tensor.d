module rk.densor.tensor;

import mir.ndslice;

template tensor(size_t[] shape, T = float)
{
    /// Create a new zero-initilized tensor.
    auto tensor()
    {
        return new Tensor!(T, shape)();
    }

    /// Create new tensor with default value initialized.
    auto tensor(T value)
    {
        return new Tensor!(T, shape)(value);
    }

    /// Create tensor from existing data.
    auto tensor(T[] data)
    {
        return new Tensor!(T, shape)(data);
    }

    /// Create tensor from RoR data (array of arrays).
    auto tensor(RoR)(RoR data)
    {
        return new Tensor!(T, shape)(data);
    }
}

class Tensor(T = float, size_t[] shape)
{
    /// make aliases
    alias SliceType = Slice!(T*, shape.length);

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
        auto tmp = slice!T(shape);
        tmp[] = value;
        this.data = tmp;
    }

    /// Create tensor from existing data.
    this(T[] data)
    {
        this.data = data.sliced(shape);
    }

    /// Create tensor from RoR data (array of arrays).
    this(RoR)(RoR data)
    {
        this.data = data.fuse.as!T.slice;
    }
}

