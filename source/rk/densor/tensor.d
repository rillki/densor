module rk.densor.tensor;

import std.array : array;
import std.format : format;
import std.traits : isArray;

import mir.ndslice;

import rk.densor.util;

/// Check if something is a tensor.
enum isTensor(T) = __traits(hasMember, T, "data") && isSlice!(typeof(T.data));

/// Create new tensor.
template tensor(size_t[] Shape, T = float)
{
    // alias InputArray = T[];
    // alias RoR = T[][];

    /// Create a new zero-initilized tensor.
    auto tensor()
    {
        return new Tensor!(T, Shape)();
    }
    
    /// Create new tensor from initialized values.
    auto tensor(T[] input)
    {
        return new Tensor!(T, Shape)(input);
    }

    /// Create tensor from existing data.
    auto tensor(SomeInput)(SomeInput input)
    {
        return new Tensor!(T, Shape)(input);
    }
}

class Tensor(T = float, size_t[] Shape)
{
    /// alias of slice type
    alias SliceType = Slice!(T*, Shape.length);

    /// slice object
    SliceType data;

    /// Create a new zero-initilized tensor.
    this()
    {
        this(0);
    }

    /// Create new tensor with default value initialized.
    this(T value)
    {
        this.data = slice!T(Shape, value);
    }

    /// Create tensor from existing data.
    this(T[] data)
    in(Shape.totalElements == data.length, 
       format!"Mismatch in the number of elements expected/given: %s != %s"(Shape.totalElements, data.length))
    {
        this.data = data.sliced(Shape);
    }

    /// Create tensor from RoR data (ndarray).
    this(RoR)(RoR data) if (isArray!RoR)
    {
        this.data = data.fuse.as!T.slice;
    }

    /// Create tensor from slice.
    this(SomeSlice)(SomeSlice slice) if (isSlice!SomeSlice)
    in(Shape.totalElements == slice.shape.totalElements, 
       format!"Mismatch in the number of elements expected/given: %s != %s"(Shape.totalElements, slice.shape.totalElements))
    {
        this.data = slice.as!T.flattened.sliced(Shape);
    }

    /// Create tensor from another tensor. 
    this(SomeTensor)(SomeTensor tensor) if (isTensor!SomeTensor)
    in(Shape.totalElements == tensor.shape.totalElements, 
       format!"Mismatch in the number of elements expected/given: %s != %s"(Shape.totalElements, tensor.shape.totalElements))
    {
        this(tensor.data);
    }

    invariant()
    {
        // import std.stdio;
        // writeln(this.data[]);
        assert(
            this.data.shape == Shape, 
            format!"Shape mismatch with expected/given: %s != %s"(Shape, this.data.shape)
        );
    }

    /// Return shape of tensor.
    auto shape() => this.data.shape;

    /// Return number of dimensions.
    auto ndim() => this.data.shape.length;
}

unittest
{
    import std.stdio : writeln;
    import std.exception : assertThrown;
    import core.exception : AssertError;

    // check if is tensor
    assert(isTensor!(Tensor!(float, [4])));

    /*  ----
        Test tensor class directly
    */
    {
        // init by default
        auto v = new Tensor!(float, [4])();
        assert(v.shape == [4]);
        assert(v.ndim == 1);
        assert(v.data[] == [0, 0, 0, 0]);
        assert(isTensor!(typeof(v)));

        // init with value
        auto vv = new Tensor!(float, [4])(1);
        assert(vv.shape == [4]);
        assert(vv.ndim == 1);
        assert(vv.data[] == [1, 1, 1, 1]);

        // init with value
        auto vvv = new Tensor!(float, [4])([1, 2, 3, 4]);
        assert(vvv.shape == [4]);
        assert(vvv.ndim == 1);
        assert(vvv.data[] == [1, 2, 3, 4]);
 
        // init with array
        auto w = new Tensor!(float, [2, 2])([1, 2, 3, 4]);
        assert(w.shape == [2, 2]);
        assert(w.ndim == 2);
        assert(w.data[] == [[1, 2], [3, 4]]);

        // init with RoR/ndarray
        auto ww = new Tensor!(float, [1, 4])([[1, 2, 3, 4]]);
        assert(ww.shape == [1, 4]);
        assert(ww.ndim == 2);
        assert(ww.data[] == [[1, 2, 3, 4]]);
        assertThrown!AssertError({ // mismatch in expected Shape and the shape of given data
            new Tensor!(float, [4, 1])([[1, 2, 3, 4]]);
        }());
    
        // init with RoR/ndarray
        auto www = new Tensor!(float, [4, 1])([[1], [2], [3], [4]]);
        assert(www.shape == [4, 1]);
        assert(www.ndim == 2);
        assert(www.data[] == [[1], [2], [3], [4]]);
        assertThrown!AssertError({ // mismatch in expected Shape and the shape of given data
            new Tensor!(float, [1, 4])([[1], [2], [3], [4]]);
        }());

        // init from the same slice type
        auto s = slice!float([2, 2], 0);
        auto x = new Tensor!(float, [2, 2])(s);
        assert(x.shape == [2, 2]);
        assert(x.ndim == 2);
        assert(x.data[] == [[0, 0], [0, 0]]);

        // init from some slice type (slice type is different, so it is reshaped)
        auto xx = new Tensor!(float, [1, 4])(s);
        assert(xx.shape == [1, 4]);
        assert(xx.ndim == 2);
        assert(xx.data[] == [[0, 0, 0, 0]]);

        // init from tensor.slice object
        auto xxx = new Tensor!(float, [1, 4])(vv.data);
        assert(xxx.shape == [1, 4]);
        assert(xxx.ndim == 2);
        assert(xxx.data[] == [[1, 1, 1, 1]]);

        // init from another tensor
        auto y = new Tensor!(float, [2, 2])(vv);
        assert(y.shape == [2, 2]);
        assert(y.ndim == 2);
        assert(y.data[] == [[1, 1], [1, 1]]);
   }

    /*  ----
        Test tensor template.
    */
    {
        // init by default
        auto v = tensor!([4])();
        assert(v.shape == [4]);
        assert(v.ndim == 1);
        assert(v.data[] == [0, 0, 0, 0]);
        assert(isTensor!(typeof(v)));

        // init with value
        auto vv = tensor!([4])(1);
        assert(vv.shape == [4]);
        assert(vv.ndim == 1);
        assert(vv.data[] == [1, 1, 1, 1]);

        // init with value
        auto vvv = tensor!([4])([1, 2, 3, 4]);
        assert(vvv.shape == [4]);
        assert(vvv.ndim == 1);
        assert(vvv.data[] == [1, 2, 3, 4]);
 
        // init with array
        auto w = tensor!([2, 2])([1, 2, 3, 4]);
        assert(w.shape == [2, 2]);
        assert(w.ndim == 2);
        assert(w.data[] == [[1, 2], [3, 4]]);

        // // init with RoR/ndarray
        auto ww = tensor!([1, 4])([[1, 2, 3, 4]]);
        assert(ww.shape == [1, 4]);
        assert(ww.ndim == 2);
        assert(ww.data[] == [[1, 2, 3, 4]]);
        assertThrown!AssertError({ // mismatch in expected Shape and the shape of given data
            new Tensor!(float, [4, 1])([[1, 2, 3, 4]]);
        }());
    
        // init with RoR/ndarray
        auto www = tensor!([4, 1])([[1], [2], [3], [4]]);
        assert(www.shape == [4, 1]);
        assert(www.ndim == 2);
        assert(www.data[] == [[1], [2], [3], [4]]);
        assertThrown!AssertError({ // mismatch in expected Shape and the shape of given data
            new Tensor!(float, [1, 4])([[1], [2], [3], [4]]);
        }());

        // init from the same slice type
        auto s = slice!float([2, 2], 0);
        auto x = tensor!([2, 2])(s);
        assert(x.shape == [2, 2]);
        assert(x.ndim == 2);
        assert(x.data[] == [[0, 0], [0, 0]]);

        // init from some slice type (slice type is different, so it is reshaped)
        auto xx = tensor!([1, 4])(s);
        assert(xx.shape == [1, 4]);
        assert(xx.ndim == 2);
        assert(xx.data[] == [[0, 0, 0, 0]]);

        // init from tensor.slice object
        auto xxx = tensor!([1, 4])(vv.data);
        assert(xxx.shape == [1, 4]);
        assert(xxx.ndim == 2);
        assert(xxx.data[] == [[1, 1, 1, 1]]);

        // init from another tensor
        auto y = tensor!([2, 2])(vv);
        assert(y.shape == [2, 2]);
        assert(y.ndim == 2);
        assert(y.data[] == [[1, 1], [1, 1]]);
    }
}

