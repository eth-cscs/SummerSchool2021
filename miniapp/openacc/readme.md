# OpenACC version of the miniapp using a single node

## Compiler support

- You may use the PGI compiler if you want to make use of the `Field` class
  operators for accessing its internal storage (recommended way).
  This is because the Cray compiler does not support the implicit attach
  operation when copying the `Field` to the GPU, so you may only copy its
  internal data and not the whole object.
- In order to use the Cray compiler, you should work with device pointers (by
  calling `device_data()`) and implement the access operators manually every
  time.
