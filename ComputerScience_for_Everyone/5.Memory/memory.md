# Memory

## Malloc
- memory allocation function
- allocates a block of memory on the heap
- returns a pointer to the first byte of the allocated memory
- if the memory allocation is successful, `malloc` returns a pointer to the memory block
- if the memory allocation is unsuccessful, `malloc` returns `NULL`

### Syntax
```c
void *malloc(size_t size);
```

- The `malloc` function takes a single argument `size` of type `size_t` which specifies the number of bytes to allocate.
- The `malloc` function returns a pointer of type `void` to the first byte of the allocated memory block.

## Free
- memory deallocation function
- deallocates the memory block previously allocated by `malloc`
- frees the memory block and makes it available for further allocation

### Syntax
```c
void free(void *ptr);
```

- The `free` function takes a single argument `ptr` of type `void` which is a pointer to the memory block to be deallocated.

## valgrind
- a tool for detecting memory leaks
- can be used to check for memory leaks in C programs
- can be used to check for memory leaks in C++ programs

### Installation
```bash
sudo apt-get install valgrind
```

### Usage
```bash
valgrind --leak-check=full ./program
```

- The `--leak-check=full` option is used to check for memory leaks in the program.
- The `./program` argument specifies the name of the program to be checked for memory leaks.

## Memory Leak
- occurs when memory is allocated but not deallocated
- can lead to memory exhaustion
- can lead to system instability
- can lead to system crashes
- can lead to security vulnerabilities
- and so on
