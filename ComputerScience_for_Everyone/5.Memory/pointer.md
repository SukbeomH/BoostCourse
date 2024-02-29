# Pointer
- A pointer is a variable that stores the memory address of another variable.
- Pointers are used to store the address of a variable, which can be used to access the value of the variable.
- Pointers are used in computer science and programming to represent memory addresses and other data.

## & Operator
- The `&` operator is used to get the memory address of a variable.
- The `&` operator is also called the address-of operator.

## * Operator
- The `*` operator is used to declare a pointer variable.
- The `*` operator is also called the dereference operator.
- The `*` operator is used to access the value of the variable that the pointer is pointing to.

## Example
```c
#include <stdio.h>

int main() {
    int x = 10;
    int *ptr = &x;

    printf("Value of x: %d\n", x);
    printf("Address of x: %p\n", &x);
    printf("Value of x using pointer: %d\n", *ptr);
    printf("Address of x using pointer: %p\n", ptr);

    return 0;
}
```

## Output
```
Value of x: 10
Address of x: 0x7ffebc3f3bfc
Value of x using pointer: 10
Address of x using pointer: 0x7ffebc3f3bfc
```

## Explanation
- In this example, we have a variable `x` with the value `10`.
- We declare a pointer variable `ptr` of type `int` and assign the memory address of `x` to it using the `&` operator.
- We then use the `*` operator to access the value of `x` using the pointer `ptr`.
- We also print the memory address of `x` using the `&` operator and the pointer `ptr`.

