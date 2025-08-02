module app;

import std.stdio;
import mir.ndslice; 

void main()
{
    auto v = slice!float(2, 2);

    writeln(typeof(v));
}
