print("화씨 온도를 섭씨 온도로 변환해주는 프로그램입니다.")
print("변환하고 싶은 화씨 온도를 입력해주세요.")

fahrenheit = float(input())

def convert_fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5 / 9

celsius = convert_fahrenheit_to_celsius(fahrenheit)

print(f"화씨 온도로 {fahrenheit:.3f}도는")
print(f"섭씨 온도로 {celsius:.3f}도 입니다.")