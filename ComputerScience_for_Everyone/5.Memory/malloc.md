# Malloc

## Introduction

The `malloc` function is used to allocate a block of memory on the heap. It is used to allocate a specified number of bytes and returns a pointer to the first byte of the allocated memory. If the memory allocation is successful, `malloc` returns a pointer to the memory block. If the memory allocation is unsuccessful, `malloc` returns `NULL`.

## Syntax

The syntax of the `malloc` function is as follows:

```c
void *malloc(size_t size);
```

- The `malloc` function takes a single argument `size` of type `size_t` which specifies the number of bytes to allocate.
- The `malloc` function returns a pointer of type `void` to the first byte of the allocated memory block.
