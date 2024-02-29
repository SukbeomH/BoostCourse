# 배열

## C에는 아래와 같은 여러 자료형이 있고, 각각의 자료형은 서로 다른 크기의 메모리를 차지합니다.
- bool: 불리언, 1바이트
- char: 문자, 1바이트
- int: 정수, 4바이트
- float: 실수, 4바이트
- long: (더 큰) 정수, 8바이트
- double: (더 큰) 실수, 8바이트
- string: 문자열, ?바이트

## 배열은 같은 자료형의 데이터를 메모리상에 연이어서 저장하고 이를 하나의 변수로 관리하기 위해 사용됩니다.

```c
int arr[5] = {1, 2, 3, 4, 5};
```

## 배열의 각 요소에 접근하기 위해서는 인덱스를 사용합니다.

## 문자열은 문자의 배열로 이해할 수 있습니다.

```c
char str[6] = "Hello";
```

- 문자열의 끝에는 널 문자(`\0`)가 포함되어야 합니다.