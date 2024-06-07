# Handling Exceptions

## 1. 예외 처리
- 프로그램을 실행하다 보면 예외가 발생하는 경우가 있다. 이때 프로그램이 비정상적으로 종료되는 것을 방지하기 위해 예외 처리를 해야 한다.
- 명시적인 예외 처리를 통해 프로그램이 비정상적으로 종료되는 것을 방지할 수 있다.

## 예상할 수 있는 예외 & 예상할 수 없는 예외
- 예상할 수 있는 예외: 파일을 읽을 때 파일이 없는 경우, 숫자를 0으로 나누는 경우 등
- 예상할 수 없는 예외: 메모리 부족, 파일 시스템에 문제 등
- 예상할 수 있는 예외는 예외 처리를 통해 프로그램이 비정상적으로 종료되는 것을 방지할 수 있지만, 예상할 수 없는 예외는 프로그램이 비정상적으로 종료될 수 있다.

### 1.1 try-except 문
- try 블록에는 예외가 발생할 수 있는 코드를 작성하고, except 블록에는 예외가 발생했을 때 실행할 코드를 작성한다.
- except 블록은 예외가 발생했을 때 실행할 코드를 작성한다.
- except 블록은 여러 개를 사용할 수 있으며, 각 예외에 따라 다른 코드를 실행할 수 있다.
- except 블록은 예외 클래스를 지정하여 특정 예외가 발생했을 때만 처리할 수 있다.
  
  ```python
  try:
      # 예외가 발생할 수 있는 코드
  except 예외 클래스 as 예외 객체:
      # 예외가 발생했을 때 실행할 코드
  ```

### 1.2 else 문
- try 블록에 예외가 발생하지 않았을 때 실행할 코드를 else 블록에 작성한다.
- else 블록은 생략할 수 있다.

  ```python
  try:
      # 예외가 발생할 수 있는 코드
  except 예외 클래스 as 예외 객체:
      # 예외가 발생했을 때 실행할 코드
  else:
      # 예외가 발생하지 않았을 때 실행할 코드
  ```

### 1.3 finally 문
- finally 블록은 예외 발생 여부와 상관없이 항상 실행된다.

  ```python
  try:
      # 예외가 발생할 수 있는 코드
  except 예외 클래스 as 예외 객체:
      # 예외가 발생했을 때 실행할 코드
  else:
      # 예외가 발생하지 않았을 때 실행할 코드
  finally:
      # 예외 발생 여부와 상관없이 항상 실행할 코드
  ```

### 1.4 예외 발생시키기
- raise 문을 사용하여 예외를 발생시킬 수 있다.
- raise 문은 예외 클래스와 예외 객체를 지정하여 예외를 발생시킬 수 있다.

  ```python
  raise 예외 클래스(예외 객체)
  ```

## 2. 사용자 정의 예외
- 사용자 정의 예외는 Exception 클래스를 상속받아 사용자가 직접 예외 클래스를 정의할 수 있다.
- 사용자 정의 예외는 raise 문을 사용하여 예외를 발생시킬 수 있다.

  ```python
  class 사용자정의예외(Exception):
      def __init__(self):
          super().__init__('사용자 정의 예외가 발생했습니다.')
  ```

## 3. 로그 처리
- 로그는 프로그램이 실행되는 동안 발생하는 이벤트를 기록하는 것이다.
- 로그는 프로그램의 실행 상태를 확인하거나 디버깅할 때 유용하다.
- 로그는 파일 또는 콘솔에 출력할 수 있다.

### 3.1 로그 레벨
- 로그 레벨은 로그의 중요도를 나타낸다.

  | 로그 레벨 | 설명 |
  |:---:|:---:|
  | DEBUG | 디버깅 목적으로 사용한다. |
  | INFO | 정보를 나타낸다. |
  | WARNING | 경고를 나타낸다. |
  | ERROR | 오류를 나타낸다. |
  | CRITICAL | 심각한 오류를 나타낸다. |

### 3.2 로그 처리기
- 로그 처리기는 로그를 출력하는 방법을 정의한다.
- 로그 처리기는 로그를 파일 또는 콘솔에 출력할 수 있다.

  ```python
  import logging

  # 로그 처리기 생성
  logger = logging.getLogger('my_logger')
  logger.setLevel(logging.DEBUG)

  # 파일 핸들러 생성
  file_handler = logging.FileHandler('my.log')
  file_handler.setLevel(logging.DEBUG)

  # 콘솔 핸들러 생성
  stream_handler = logging.StreamHandler()
  stream_handler.setLevel(logging.DEBUG)

  # 로그 포맷 지정
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  file_handler.setFormatter(formatter)
  stream_handler.setFormatter(formatter)

  # 로그 처리기에 핸들러 추가
  logger.addHandler(file_handler)
  logger.addHandler(stream_handler)

  # 로그 출력
  logger.debug('디버깅 메시지')
  logger.info('정보 메시지')
  logger.warning('경고 메시지')
  logger.error('오류 메시지')
  logger.critical('심각한 오류 메시지')
  ```

### 3.3 로그 파일
- 로그 파일은 로그를 기록하는 파일이다.

  ```python
  import logging

  # 로그 파일 생성
  logging.basicConfig(filename='my.log', level=logging.DEBUG)

  # 로그 출력
  logging.debug('디버깅 메시지')
  logging.info('정보 메시지')
  logging.warning('경고 메시지')
  logging.error('오류 메시지')
  logging.critical('심각한 오류 메시지')
  ```

### 3.4 로그 설정 파일
- 로그 설정 파일은 로그 처리기의 설정을 파일로 관리한다.

  ```python
  import logging
  import logging.config

  # 로그 설정 파일 읽기
  logging.config.fileConfig('logging.conf')

  # 로그 출력
  logging.debug('디버깅 메시지')
  logging.info('정보 메시지')
  logging.warning('경고 메시지')
  logging.error('오류 메시지')
  logging.critical('심각한 오류 메시지')
  ```

  ```python
  [loggers]
  keys=root

  [handlers]
  keys=consoleHandler

  [formatters]
  keys=consoleFormatter

  [logger_root]
  level=DEBUG
  handlers=consoleHandler

  [handler_consoleHandler]
  class=StreamHandler
  level=DEBUG
  formatter=consoleFormatter
  args=(sys.stdout,)

  [formatter_consoleFormatter]
  format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
  ```

## 4. 참고
- https://docs.python.org/3/library/exceptions.html
- https://docs.python.org/3/library/logging.html
- https://docs.python.org/3/howto/logging.html



