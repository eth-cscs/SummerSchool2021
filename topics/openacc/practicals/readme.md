# OpenACC practical exercises

- Exercises are tested with the PGI compiler.
- Compile each example with `make`
- Run as `srun --reserv=course -Cgpu <exec-name> [ARRAY_SIZE]`

`ARRAY_SIZE` is always a power of 2, so putting 10 would mean an array of 1024 elements.
