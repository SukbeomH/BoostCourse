# Address of Memory
- Memory is a large array of bytes, each with its own address.
- Each byte has its own address, which is a unique number.
- The address of a byte is the number of the first byte in the memory.

## Example
```c
#include <stdio.h>

int main() {
    int num = 10;
    printf("%p\n", &num);
    return 0;
}
```
- The `&` operator is used to get the address of a variable.
- The `%p` format specifier is used to print the address of a variable.

## What is & Operator?
- The `&` operator is used to get the address of a variable.
- The `&` operator is also called the address-of operator.

## What is %p Format Specifier?
- The `%p` format specifier is used to print the address of a variable.
- The `%p` format specifier is used with the `printf` function to print the address of a variable.

## Output
```
0x7ffebc3f3bfc
```
- The output is the memory address of the variable `num`.

## Explanation
- In this example, we have a variable `num` with the value `10`.
- We use the `&` operator to get the memory address of `num`.
- We then print the memory address of `num` using the `%p` format specifier.

