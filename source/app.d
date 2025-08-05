module app;

import std.stdio;
import mir.ndslice; 

import rk.densor;

void main()
{
    auto a = slice!float([2, 2], 0);
    a.writeln;

    writeln(a.flattened.sliced([1, 4]));
    writeln([1, 2, 3, 4].sliced([4, 1]));



    // auto v = new Tensor!(float, [2, 2])([1, 2, 3, 4]);
    // auto w = new Tensor!(float, [2, 2])(v);
    // v.data[] = 12;
    // writeln(v.data);

    // auto x = tensor!([2, 2])(-3);
    // auto y = tensor!([2, 2])([1, 2, 3, 4]);
    // auto z = tensor!([2, 2])(x);
    // x.data[] = 1;
    // writeln(x.data, "\n", z.data);
}
