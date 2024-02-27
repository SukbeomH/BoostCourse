# argument
- argument is a value that is passed to a function when it is called.
- The function uses the argument to perform its task.
- The argument is specified inside the parentheses of the function call.

# parameter
- parameter is a variable that is used to define a function.
- The parameter is the variable listed inside the parentheses in the function definition.
- When the function is called, the argument is the data you pass into the function's parameters.

# Example
```c
#include <stdio.h>

void printNumber(int num) {
    printf("%d\n", num);
}

int main() {
    printNumber(10);
    return 0;
}
```
