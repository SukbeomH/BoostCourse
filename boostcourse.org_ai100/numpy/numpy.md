# NumPy

## NumPy란?
- **Numerical Python**의 약자로, 파이썬에서 수치 계산을 위해 사용되는 라이브러리.
- 다차원 배열을 효과적으로 처리할 수 있으며, 선형대수와 관련된 다양한 기능을 제공한다.

- NumPy는 다음과 같은 기능을 제공한다.
    - **다차원 배열**을 효과적으로 처리할 수 있는 기능을 제공한다.
    - **선형대수**와 관련된 다양한 기능을 제공한다.
    - C/C++ 및 포트란과 같은 **저수준 언어로 작성된 코드를 연결**할 수 있는 도구를 제공한다.
    - 데이터 처리에 유용한 다양한 기능을 제공한다.

사실상 Matrix, Vector 연산을 위한 array 연산의 표준이다.

## NumPy의 특징
- 일반 List에 비해 빠르고 메모리를 효율적으로 사용한다.
- 반복문 없이 데이터 배열에 대한 처리를 지원한다.
    - 이를 **배열 지향 프로그래밍**이라고 한다.
- 배열 연산을 통해 선형 대수 연산을 수행할 수 있다.

#### 배열지향 프로그래밍
배열 지향 프로그래밍은 배열 처리에 대한 연산을 반복문 없이 처리하는 방식을 말한다. 
이를 통해 코드의 가독성이 높아지고, 코드 실행 속도가 빨라진다.

```python
# 배열 지향 프로그래밍 예시
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])

result = a + b

print(result) # [3 5 7 9]
```

위 코드에서 a와 b는 각각 1차원 배열이다.
a와 b의 각 요소를 더한 결과를 result에 저장하고 있다.
이때, **반복문을 사용하지 않고** 각 요소를 더한 결과를 얻을 수 있다.

## NumPy 배열 생성하기
NumPy 배열은 `np.array()` 함수를 사용하여 생성할 수 있다.

```python
import numpy as np

# 1차원 배열 생성
a = np.array([1, 2, 3, 4, 5])
print(a) # [1 2 3 4 5]

# 2차원 배열 생성
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)
# [[1 2 3]
#  [4 5 6]]
```

- `np.array()` 함수는 파이썬의 리스트를 입력받아 NumPy 배열로 변환한다.
- ndarray 클래스는 **N-dimensional Array**의 약자로, NumPy의 기본 자료구조이다.
  - 기존 List와 ndarray의 차이점은 ndarray는 **같은 자료형**만 저장할 수 있다는 점이다.
  - ndarray는 배열의 차원, 형태, 데이터 타입을 나타내는 shape, dtype 속성을 가진다.
    - **shape** 속성은 배열의 차원과 크기를 나타내는 튜플이다.
    - **dtype** 속성은 배열의 데이터 타입을 나타낸다.
  
내부적으로 C로 구현되어 있어 빠르고 메모리를 효율적으로 사용한다.
- List의 경우 각 **메모리에 주소값을 저장**하고 있어 메모리를 많이 차지하고, 연산이 느리다.
- ndarray는 **연속된 메모리에 데이터를 곧바로 저장**하므로 빠르고 메모리를 효율적으로 사용한다.
  - 메모리의 크기도 모두 동일하기 때문에 연산 속도가 빠르다.
  - List는 각 요소의 크기가 다르기 때문에 메모리를 효율적으로 사용하지 못한다.

## Array Shape
- NumPy 배열의 차원과 크기를 나타내는 속성이다.

- **0차원 배열**
    - 하나의 숫자로 이루어진 배열이다.
    - shape 속성은 `()` 형태로 표시된다.
    - 0차원 배열은 **Scalar**라고도 한다.
    - `np.array(1)`의 shape 속성은 `()`이다.

- **1차원 배열**
    - 배열의 크기만 나타낸다.
    - shape 속성은 `(배열의 크기,)` 형태로 표시된다.
    - 1차원 배열은 **Vector**라고도 한다.
    - `np.array([1, 2, 3, 4, 5])`의 shape 속성은 `(5,)`이다.

- **2차원 배열**
    - **Matrix**라고도 한다.
    - shape 속성은 `(행, 열)` 형태로 표시된다.
    - `np.array([[1, 2, 3], [4, 5, 6]])`의 shape 속성은 `(2, 3)`이다.

- **N차원 배열**
    - shape 속성은 `(차원1의 크기, 차원2의 크기, ..., 차원n의 크기)` 형태로 표시된다.
    - `np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])`의 shape 속성은 `(2, 2, 2)`이다.
    - 3차원 배열은 **Tensor**라고도 한다.
        - Tensor는 딥러닝에서 많이 사용되는 자료구조이다.
    - N차원 배열은 **다차원 배열**이라고도 한다.

## NumPy 배열의 데이터 타입 (dtype)
- NumPy 배열은 **같은 데이터 타입**만 저장할 수 있다.
- 배열 생성 시 dtype 속성을 사용하여 데이터 타입을 지정할 수 있다.
- dtype 속성을 사용하지 않으면 NumPy는 주어진 데이터를 저장할 수 있는 자료형을 스스로 선택한다.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a.dtype) # int64

b = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
print(b.dtype) # float64

c = np.array([1, 2, 3, 4.1, 5.2])
print(c.dtype) # float64
```

- NumPy 배열의 데이터 타입은 `dtype` 속성을 통해 확인할 수 있다.
    - 파이썬의 기본 자료형과 달리 NumPy는 **데이터 타입의 크기를 명확하게 지정**한다.
    - `int64`는 64비트 정수형을 의미한다.
    - `float64`는 64비트 부동소수점형을 의미한다.
    - `int32`, `float32`, `int16`, `float16` 등 다양한 데이터 타입이 있다.
- 데이터 타입을 지정하려면 `dtype` 속성을 사용한다.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5], dtype='float32')
print(a.dtype) # float32

b = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype='int32')
print(b.dtype) # int32
```
## Shape Handling
- NumPy 배열의 shape 속성을 변경할 수 있다.
- `reshape()` 함수를 사용하여 배열의 shape 속성을 변경할 수 있다.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])
print(a.shape) # (6,)
print(a) # [1 2 3 4 5 6]

b = a.reshape(2, 3)
print(b.shape) # (2, 3)
print(b)
# [[1 2 3]
#  [4 5 6]]
```

- `reshape()` 함수는 배열의 shape 속성을 변경한다.
- `reshape(2, 3)`은 2행 3열로 배열의 shape 속성을 변경한다.
- `reshape()` 함수는 원본 배열의 요소 개수와 변경하려는 배열의 요소 개수가 일치해야 한다.

## Flattening
- 다차원 배열을 1차원 배열로 변환하는 것을 **평탄화**라고 한다.
- `flatten()` 함수를 사용하여 다차원 배열을 1차원 배열로 변환할 수 있다.

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape) # (2, 3)

b = a.flatten()
print(b.shape) # (6,)
print(b) # [1 2 3 4 5 6]
```

- `flatten()` 함수는 원본 배열의 **복사본**을 반환한다.

평탄화를 하는 이유는 **데이터를 일렬로 나열**하여 저장하고 처리하기 위함이다.
- 딥러닝에서는 **평탄화된 데이터**를 입력으로 사용한다.
- 이미지 데이터는 2차원 배열로 표현되지만, 딥러닝에서는 1차원 배열로 변환하여 사용한다.
- 평탄화된 데이터를 사용하면 **데이터 처리 속도가 빨라지고, 메모리를 효율적으로 사용**할 수 있다.

다른 방법으로는 `reshape(-1)`을 사용할 수 있다.
- `reshape(-1)`은 다차원 배열을 1차원 배열로 변환한다.
- `reshape(-1, 1)`은 1차원 배열을 열 벡터로 변환한다.
- `reshape(1, -1)`은 1차원 배열을 행 벡터로 변환한다.

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape) # (2, 3)

b = a.reshape(-1)
print(b.shape) # (6,)
print(b) # [1 2 3 4 5 6]

c = b.reshape(-1, 1)
print(c.shape) # (6, 1)
print(c)
# [[1]
#  [2]
#  [3]
#  [4]
#  [5]
#  [6]]

d = b.reshape(1, -1)
print(d.shape) # (1, 6)
print(d)
# [[1 2 3 4 5 6]]
```

## Indexing
- NumPy 배열의 요소에 접근하는 방법을 **인덱싱**이라고 한다.
- NumPy 배열은 0부터 시작하는 인덱스를 사용한다.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a[0]) # 1
print(a[1]) # 2
print(a[-1]) # 5
```

기본적으로 파이썬의 리스트와 동일하게 인덱싱을 사용할 수 있다.

## Slicing
- NumPy 배열의 일부를 추출하는 방법을 **슬라이싱**이라고 한다.
- 슬라이싱은 `:`을 사용하여 범위를 지정한다.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a[0:2]) # [1 2]
print(a[1:]) # [2 3 4 5]
print(a[:3]) # [1 2 3]
```

파이썬의 리스트와 동일하게 슬라이싱을 사용할 수 있다.
추가적으로 리스트와 달리 NumPy 배열은 **다차원 배열**이므로 **다차원 슬라이싱**도 가능하다.

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a[0:2, 0:2])
# [[1 2]
#  [4 5]]
```

- `a[0:2, 0:2]`는 0행부터 1행, 0열부터 1열까지의 요소를 추출한다.

## Boolean Indexing
- NumPy 배열에서 조건을 사용하여 요소를 추출하는 방법을 **Boolean Indexing**이라고 한다.
- 조건을 사용하여 True, False 값을 반환하고, True에 해당하는 요소만 추출한다.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a[a > 2]) # [3 4 5]
```

- `a > 2`는 배열 a의 요소가 2보다 큰지 확인하여 True, False 값을 반환한다.
- `a[a > 2]`는 True에 해당하는 요소만 추출한다.
- Boolean Indexing은 조건을 사용하여 요소를 추출할 때 유용하다.
- 조건을 사용하여 요소를 추출할 때는 **반드시 괄호**를 사용해야 한다.
- 괄호를 사용하지 않으면 에러가 발생한다.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a[(a > 2) & (a < 5)]) # [3 4]
```

## Creation Functions
- NumPy 배열을 생성하는 다양한 함수가 있다.

### `np.zeros()`
- 지정한 shape의 배열을 생성하고, 모든 요소를 0으로 초기화한다.

```python
import numpy as np

a = np.zeros((2, 3))
print(a)
# [[0. 0. 0.]
#  [0. 0. 0.]]
```

### `np.ones()`
- 지정한 shape의 배열을 생성하고, 모든 요소를 1로 초기화한다.

```python
import numpy as np

a = np.ones((2, 3))
print(a)
# [[1. 1. 1.]
#  [1. 1. 1.]]
```

### `np.empty()`
- 지정한 shape의 배열을 생성하고, 모든 요소를 초기화하지 않는다.

```python
import numpy as np

a = np.empty((2, 3))
print(a)
# [[1. 1. 1.]
#  [1. 1. 1.]]
```

- `np.empty()` 함수는 배열을 생성할 때 **메모리를 할당**하지만, **초기화하지 않는다**.
    - 초기화하지 않기 때문에 배열의 요소는 **이전에 메모리에 저장된 값**이다.


### `np.full()`
- 지정한 shape의 배열을 생성하고, 모든 요소를 지정한 값으로 초기화한다.

```python
import numpy as np

a = np.full((2, 3), 7)
print(a)
# [[7 7 7]
#  [7 7 7]]
```

### `np.eye()`
- 지정한 크기의 단위 행렬을 생성한다.
- 단위 행렬은 주대각선의 요소가 1이고, 나머지 요소가 0인 행렬이다.

```python
import numpy as np

a = np.eye(3)
print(a)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

대각선의 시작점을 변경하려면 `k` 인자를 사용한다.

```python
import numpy as np

a = np.eye(3, k=1)

print(a)
# [[0. 1. 0.]
#  [0. 0. 1.]
#  [0. 0. 0.]]
```

### `np.identity()`
- 지정한 크기의 단위 행렬을 생성한다.
- `np.eye()` 함수와 비슷하지만, 시작점을 변경할 수 없다.

```python
import numpy as np

a = np.identity(3)
print(a)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

### `np.diagonal()`
- 주어진 배열의 대각선 요소를 추출한다.

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(np.diagonal(a)) # [1 5 9]
```

- `np.diagonal()` 함수는 주어진 배열의 대각선 요소를 추출한다.
- `np.diagonal()` 함수는 **대각선의 시작점을 변경**할 수 있다.

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(np.diagonal(a, offset=1)) # [2 6]
```

### `np.random.random()`
- 지정한 shape의 배열을 생성하고, 모든 요소를 랜덤한 값으로 초기화한다.

```python
import numpy as np

a = np.random.random((2, 3))
print(a)
# [[0.5488135  0.71518937 0.60276338]
#  [0.54488318 0.4236548  0.64589411]]
```

- `np.random.random()` 함수는 0과 1 사이의 랜덤한 값으로 배열을 초기화한다.

### `np.random.uniform()`
- 균등 분포에서 랜덤한 값으로 배열을 생성한다.

```python
import numpy as np

a = np.random.uniform(0, 1, (2, 3))
print(a)
# [[0.96366276 0.38344152 0.79172504]
#  [0.52889492 0.56804456 0.92559664]]
```

### `np.arange()`
- 지정한 범위의 배열을 생성한다.

```python
import numpy as np

a = np.arange(10)
print(a) # [0 1 2 3 4 5 6 7 8 9]

b = np.arange(1, 10, 2)
print(b) # [1 3 5 7 9]
```

- `np.arange(10)`은 0부터 9까지의 배열을 생성한다.
- `np.arange(1, 10, 2)`는 1부터 9까지 2씩 증가하는 배열을 생성한다.

### `np.linspace()`
- 지정한 범위의 배열을 생성한다.

```python
import numpy as np

a = np.linspace(1, 10, 5)
print(a) # [ 1.    3.25  5.5   7.75 10.  ]
```

- `np.linspace(1, 10, 5)`는 1부터 10까지 5개의 요소를 가진 배열을 생성한다.
- `np.linspace()` 함수는 **시작값, 끝값, 요소 개수**를 인자로 받는다.
- `np.linspace()` 함수는 **시작값과 끝값을 포함**하여 요소 개수만큼 균등하게 나눈 배열을 생성한다.
- `np.linspace()` 함수는 **범위를 나누는 간격을 자동으로 계산**한다.

## Operations Functions
- NumPy 배열을 다루는 다양한 함수가 있다.
- NumPy 배열은 **요소별 연산**을 지원한다.
    - **요소별 연산**은 **배열의 크기가 동일**해야 한다.
    - **요소별 연산**은 **배열의 형태가 동일**해야 한다.
    - **요소별 연산**은 **배열의 데이터 타입이 동일**해야 한다.

#### 기본 연산
- `np.add()`: 덧셈
- `np.subtract()`: 뺄셈
- `np.multiply()`: 곱셈
- `np.divide()`: 나눗셈
- `np.power()`: 제곱

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([2, 3, 4, 5, 6])

print(np.add(a, b)) # [ 3  5  7  9 11]
print(np.subtract(a, b)) # [-1 -1 -1 -1 -1]
print(np.multiply(a, b)) # [ 2  6 12 20 30]
print(np.divide(a, b)) # [0.5  0.66666667  0.75  0.8   0.83333333]
print(np.power(a, b)) # [1  8  81  1024 15625]
```

**Axis 기반 연산**
- NumPy 배열은 **다차원 배열**이므로 **축(axis) 기반 연산**을 지원한다.
- **축(axis) 기반 연산**은 **특정 축을 기준**으로 연산을 수행한다.

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print(np.sum(a)) # 21
print(np.sum(a, axis=0)) # [5 7 9]
print(np.sum(a, axis=1)) # [ 6 15]
```

- `np.sum(a)`은 배열 a의 모든 요소의 합을 반환한다.
- `np.sum(a, axis=0)`은 배열 a의 열을 기준으로 합을 반환한다.
- `np.sum(a, axis=1)`은 배열 a의 행을 기준으로 합을 반환한다.
- **axis=0**은 **열을 기준**으로 연산을 수행한다.
- **axis=1**은 **행을 기준**으로 연산을 수행한다.
- **axis=None**은 **모든 요소**를 기준으로 연산을 수행한다.

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print(np.mean(a)) # 3.5
print(np.mean(a, axis=0)) # [2.5 3.5 4.5]
print(np.mean(a, axis=1)) # [2. 5.]
```

- `np.mean(a)`은 배열 a의 모든 요소의 평균을 반환한다.
- `np.mean(a, axis=0)`은 배열 a의 열을 기준으로 평균을 반환한다.
- `np.mean(a, axis=1)`은 배열 a의 행을 기준으로 평균을 반환한다.

#### exponentials and logarithms
- `np.exp()`: 지수 함수
- `np.log()`: 자연 로그 함수

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])

print(np.exp(a)) # [  2.71828183   7.3890561   20.08553692  54.59815003 148.4131591 ]
print(np.log(a)) # [0.         0.69314718 1.09861229 1.38629436 1.60943791]
```

- `np.exp(a)`는 배열 a의 요소에 대한 지수 함수를 계산한다.
- `np.log(a)`는 배열 a의 요소에 대한 자연 로그 함수를 계산한다.

#### Concatenation
- `np.concatenate()`: 배열을 연결한다.

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.concatenate([a, b])) # [1 2 3 4 5 6]
```

- `np.concatenate([a, b])`는 배열 a와 b를 연결한다.
- `np.concatenate()` 함수는 **axis 인자**를 사용하여 **축을 지정**할 수 있다.

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])

print(np.concatenate([a, a], axis=0))
# [[1 2]
#  [3 4]
#  [1 2]
#  [3 4]]

print(np.concatenate([a, a], axis=1))
# [[1 2 1 2]
#  [3 4 3 4]]
```

- `np.hstack()`: 수평으로 배열을 연결한다.
- `np.vstack()`: 수직으로 배열을 연결한다.

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print(np.hstack([a, b]))
# [[1 2 5 6]
#  [3 4 7 8]]

print(np.vstack([a, b]))
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]
```

#### `np.newaxis()`
- 배열에 새로운 축을 추가한다.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a.shape) # (5,)
print(a[np.newaxis, :].shape) # (1, 5)
print(a[:, np.newaxis].shape) # (5, 1)
```

- `a[np.newaxis, :]`는 배열 a에 새로운 행을 추가한다.
- `a[:, np.newaxis]`는 배열 a에 새로운 열을 추가한다.
- `np.newaxis`는 **축을 추가**하는 역할을 한다.
- `np.newaxis`는 **차원을 증가**시키는 역할을 한다.

## Broadcasting
```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([1, 2, 3])

print(a + b)
# [[2 4 6]
#  [5 7 9]]
```

- `a + b`는 배열 a의 각 행에 배열 b를 더한다.
- **Broadcasting**은 **배열의 형태가 다른 경우**에도 **요소별 연산**을 수행할 수 있다.

## transpose

전치 행렬을 구하는 방법은 **배열의 T 속성**을 사용하거나 **transpose() 함수**를 사용한다.

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

print(a.T)
# [[1 4]
#  [2 5]
#  [3 6]]

print(a.transpose())
# [[1 4]
#  [2 5]
#  [3 6]]
```

#### dot product
행렬 곱셈을 구하는 방법은 **dot() 함수**를 사용한다.

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print(np.dot(a, b))
# [[19 22]
#  [43 50]]
```

- `np.dot(a, b)`는 행렬 a와 b의 행렬 곱셈을 수행한다.
- **행렬 곱셈**은 **행렬 a의 열과 행렬 b의 행이 같아야** 한다.

#### inverse
역행렬을 구하는 방법은 **inv() 함수**를 사용한다.

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])

print(np.linalg.inv(a))
# [[-2.   1. ]
#  [ 1.5 -0.5]]
```

- `np.linalg.inv(a)`는 행렬 a의 역행렬을 구한다.
    - **역행렬**은 **원래 행렬과 곱했을 때 단위 행렬**이 되는 행렬이다.
    - **역행렬**은 **정방 행렬**에만 존재한다.
    - **역행렬**은 **행렬식(determinant)**이 0이 아닌 경우에만 존재한다.
    - **역행렬**은 **행렬의 곱셈**을 **나눗셈**으로 계산할 때 사용한다.
    - **역행렬**은 **행렬의 크기가 커질수록 계산 비용이 많이 든다**.
    - **역행렬**은 **행렬의 크기가 커질수록 정확도가 떨어진다**.

#### determinant
행렬식을 구하는 방법은 **det() 함수**를 사용한다.

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])

print(np.linalg.det(a))
# -2.0000000000000004
```

**행렬식**이란 **정방 행렬**에 대해 **스칼라 값을 반환**하는 함수이다.

## Performance Comparison
- NumPy 배열은 **반복문 없이** 배열에 대한 처리를 지원한다.
- **반복문 없이** 배열에 대한 처리를 하면 **코드 실행 속도가 빨라진다**.

```python
import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

start = time.time()
c = np.dot(a, b)
end = time.time()

print(c)
print(end - start)
```

- `np.random.rand(1000000)`은 100만 개의 랜덤한 값을 가진 배열을 생성한다.
- `np.dot(a, b)`는 두 배열의 행렬 곱셈을 수행한다.
- **반복문 없이** 배열에 대한 처리를 하면 **코드 실행 속도가 빨라진다**.

## Comparison 
- NumPy 배열은 **요소별 비교**를 지원한다.
- **요소별 비교**는 **배열의 크기가 동일**해야 한다.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([2, 3, 4, 5, 6])

print(a == b) # [False False False False False]
print(a > b) # [False False False False False]
print(a < b) # [ True  True  True  True  True]
```

#### all & any
- `np.all()`: 모든 요소가 True인지 확인한다.
- `np.any()`: 하나 이상의 요소가 True인지 확인한다.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([2, 3, 4, 5, 6])

print(np.all(a == b)) # False
print(np.any(a == b)) # True
```

- `np.all(a == b)`는 배열 a와 b의 모든 요소가 같은지 확인한다.
- `np.any(a == b)`는 배열 a와 b의 요소 중 하나라도 같은지 확인한다.

#### logical_and & logical_or
- `np.logical_and()`: 논리 AND 연산을 수행한다.
- `np.logical_or()`: 논리 OR 연산을 수행한다.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([2, 3, 4, 5, 6])

print(np.logical_and(a > 2, a < 5)) # [False False  True  True False]
print(np.logical_or(a > 2, a < 5)) # [ True  True  True  True  True]
```

#### logical_not
- `np.logical_not()`: 논리 NOT 연산을 수행한다.

#### where, isfinite, isinf, isnan ...
- `np.where()`: 조건에 따라 요소를 선택한다.
- `np.isfinite()`: 유한한 수인지 확인한다.
- `np.isinf()`: 무한한 수인지 확인한다.
- `np.isnan()`: NaN인지 확인한다.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])

print(np.where(a > 2)) # (array([2, 3, 4]),)
print(np.isfinite(a)) # [ True  True  True  True  True]
print(np.isinf(a)) # [False False False False False]
print(np.isnan(a)) # [False False False False False]
```

- `np.where(a > 2)`는 배열 a에서 **조건 a > 2**를 만족하는 요소의 인덱스를 반환한다.
- `np.isfinite(a)`는 배열 a의 요소가 **유한한 수**인지 확인한다.
- `np.isinf(a)`는 배열 a의 요소가 **무한한 수**인지 확인한다.
- `np.isnan(a)`는 배열 a의 요소가 **NaN**인지 확인한다.
- **NaN**은 **Not a Number**의 약자로 **숫자가 아닌 값을 나타낸다**.

## Statistics
- NumPy 배열은 **통계 함수**를 지원한다.
- **통계 함수**는 **배열의 요소에 대한 통계량**을 계산한다.

#### mean, median, std, var
- `np.mean()`: 평균을 계산한다.
- `np.median()`: 중앙값을 계산한다.
- `np.std()`: 표준편차를 계산한다.
- `np.var()`: 분산을 계산한다.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])

print(np.mean(a)) # 3.0
print(np.median(a)) # 3.0
print(np.std(a)) # 1.4142135623730951
print(np.var(a)) # 2.0
```

- `np.mean(a)`은 배열 a의 **평균**을 계산한다.
- `np.median(a)`은 배열 a의 **중앙값**을 계산한다.
- `np.std(a)`은 배열 a의 **표준편차**를 계산한다.
- `np.var(a)`은 배열 a의 **분산**을 계산한다.


- **평균**은 **모든 요소의 합을 요소 개수로 나눈 값**이다.
- **중앙값**은 **요소를 크기 순서대로 나열했을 때 중앙에 위치한 값**이다.
- **표준편차**는 **평균으로부터 얼마나 떨어져 있는지**를 나타내는 값이다.
- **분산**은 **평균으로부터 얼마나 떨어져 있는지**를 나타내는 값이다.


## Sorting
- NumPy 배열은 **정렬**을 지원한다.
- **정렬**은 **배열의 요소를 크기 순서대로 나열**하는 것이다.

**`np.sort()`**
```python
import numpy as np

a = np.array([3, 1, 2, 4, 5])

print(np.sort(a)) # [1 2 3 4 5]
```

- `np.sort(a)`는 배열 a의 요소를 **크기 순서대로 정렬**한다.
- **정렬**은 **기본적으로 오름차순**으로 정렬한다.

**`np.argsort()`**
```python
import numpy as np

a = np.array([3, 1, 2, 4, 5])

print(np.argsort(a)) # [1 2 0 3 4]
```

- `np.argsort(a)`는 배열 a의 요소를 **정렬한 후의 인덱스**를 반환한다.
- **정렬한 후의 인덱스**는 **원래 배열의 요소를 정렬한 순서**를 나타낸다.

**`np.argmax()` & `np.argmin()`**
```python
import numpy as np

a = np.array([3, 1, 2, 4, 5])

print(np.argmax(a)) # 4
print(np.argmin(a)) # 1
```

- `np.argmax(a)`는 배열 a의 **최댓값의 인덱스**를 반환한다.
- `np.argmin(a)`는 배열 a의 **최솟값의 인덱스**를 반환한다.

## Filtering
- NumPy 배열은 **필터링**을 지원한다.
- **필터링**은 **조건을 사용하여 요소를 추출**하는 것이다.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])

print(a[a > 2]) # [3 4 5]
```

- `a[a > 2]`는 배열 a에서 **조건 a > 2**를 만족하는 요소만 추출한다.
- **필터링**은 **조건을 사용하여 요소를 추출**할 때 유용하다.

## Save & Load
- NumPy 배열은 **파일로 저장**하고 **불러올** 수 있다.
- **파일로 저장**하면 **나중에 사용**할 수 있다.

**`np.save()` & `np.load()`**
```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])

np.save('a.npy', a)

b = np.load('a.npy')

print(b) # [1 2 3 4 5]
```

- `np.save('a.npy', a)`는 배열 a를 **a.npy 파일**로 저장한다.
- `np.load('a.npy')`는 **a.npy 파일**을 불러와 배열 b에 저장한다.

**`np.savez()` & `np.load()`**
```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])

np.savez('a.npz', a=a)

b = np.load('a.npz')

print(b['a']) # [1 2 3 4 5]
```

- `np.savez('a.npz', a=a)`는 배열 a를 **a.npz 파일**로 저장한다.
- `np.load('a.npz')`는 **a.npz 파일**을 불러와 배열 b에 저장한다.

npy 파일은 **하나의 배열**을 저장하고, npz 파일은 **여러 배열**을 저장한다.

