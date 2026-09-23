# File I/O

`bartorch.io` reads and writes BART's CFL format: a `.hdr` text header with
the dimensions and a `.cfl` file of complex64 values in Fortran order.  Arrays
are NumPy arrays in BART's dimension order, the reverse of a C-order tensor
shape, so `array.T` converts between the two; {doc}`../guides/user/conventions`
shows the round trip.

```{eval-rst}
.. currentmodule:: bartorch.io
```

| Object | Description |
| --- | --- |
| {obj}`~bartorch.io.readcfl` | Read `name.hdr` and `name.cfl` into a NumPy array in BART's dimension order |
| {obj}`~bartorch.io.writecfl` | Write a NumPy array in BART's dimension order as `name.hdr` and `name.cfl` |
