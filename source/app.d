module app;

import std.stdio;
import mir.ndslice; 

import rk.densor;

void main()
{
    auto v = new Tensor!(float, [2, 2])(42);
    auto w = new Tensor!(float, [2, 2])(v);
    v.data[] = 12;
    writeln(w.data);

    auto x = tensor!([2, 2])(-3);
    auto y = tensor!([2, 2])([1, 2, 3, 4]);
    auto z = tensor!([2, 2])(x);
    x.data[] = 1;
    writeln(x.data, "\n", z.data);
}
