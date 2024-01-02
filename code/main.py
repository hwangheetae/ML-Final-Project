import joblib
from my_linear_regression import MYLinearRegression
import numpy as np
loaded_model = joblib.load('my_model.pkl')

def predict_birth_rate(model, death, death_rate, divorce, divorce_rate, marriage, marriage_rate):
    input_features = np.array([[death, death_rate, divorce, divorce_rate, marriage, marriage_rate]])
    predicted_birth_rate = model.predict(input_features)

    return predicted_birth_rate[0]


death = float(input("사망자 수를 입력하세요: "))
death_rate = float(input("사망률을 입력하세요: "))
divorce = float(input("이혼 건수를 입력하세요: "))
divorce_rate = float(input("이혼률을 입력하세요: "))
marriage = float(input("결혼 건수를 입력하세요: "))
marriage_rate = float(input("결혼률을 입력하세요: "))

predicted_birth_rate = predict_birth_rate(loaded_model, death, death_rate, divorce, divorce_rate, marriage, marriage_rate)
print(f"예측된 출산율: {predicted_birth_rate}")
