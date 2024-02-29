# String as a sequence of characters 

**Objective:** To understand the concept of string as a sequence of characters.

char *str = "Hello World!";
printf("%s", str);

- In C, a string is represented as a sequence of characters.
- A string is a sequence of characters terminated by a null character `'\0'`.
- A string is represented using the `char` data type.
- A string can be represented using the `char` array or the `char` pointer.
- A string can be printed using the `%s` format specifier with the `printf` function.

## Example
```c
#include <stdio.h>

int main() {
    char str1[] = "Hello World!";
    char *str2 = "Hello World!";
    
    printf("%s\n", str1);
    printf("%s\n", str2);
    
    return 0;
}
```

## Output
```
Hello World!
Hello World!
```

## Explanation
- In this example, we have two strings `str1` and `str2` with the value `"Hello World!"`.
- The first string `str1` is represented using the `char` array.
- The second string `str2` is represented using the `char` pointer.
- We print both strings using the `%s` format specifier with the `printf` function.

## Summary
- A string is a sequence of characters terminated by a null character `'\0'`.
- A string is represented using the `char` data type.
- A string can be represented using the `char` array or the `char` pointer.
- A string can be printed using the `%s` format specifier with the `printf` function.
