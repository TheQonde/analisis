# Definir las probabilidades condicionales de las probabilidades de COVID-19
probabilidades = {
    'Fatiga': 0.75,
    'Tos seca': 0.80,
    'Dificultad para respirar': 0.90,
    'Dolor de garganta': 0.65,
    'Dolor de cabeza': 0.20,
    'Dolor en el cuerpo': 0.30,
    'Escalofríos': 0.40,
    'Secresión nasal': 0.50,
    'Pérdida del sentido del olfato': 0.70,
    'Fiebre': 0.80,
    'Dolor de pecho': 0.70
}

# Definir la prevalencia del COVID-19 en la población
p_infectado = 0.05

# Definir los síntomas observados en el paciente
sintomas = ['Fatiga', 'Tos seca', 'Dificultad para respirar', 'Dolor de garganta', 'Dolor de cabeza',
            'Dolor en el cuerpo', 'Escalofríos', 'Secresión nasal', 'Pérdida del sentido del olfato', 'Fiebre',
            'Dolor de pecho']


def calcular_probabilidad_covid(sintomas):
    p_sintomas_infectado = 1.0
    for sintoma in sintomas:
        if sintoma in probabilidades:
            p_sintomas_infectado *= probabilidades[sintoma]
        else:
            raise ValueError(f"Síntoma no válido: {sintoma}")

    p_sintomas_no_infectado = 0.05 ** len(sintomas)

    p_sintomas = p_sintomas_infectado * p_infectado + p_sintomas_no_infectado * (1 - p_infectado)

    p_infectado_sintomas = p_sintomas_infectado * p_infectado / p_sintomas

    return p_infectado_sintomas


try:
    p_infectado_sintomas = calcular_probabilidad_covid(sintomas)
    print("La probabilidad de que el paciente tenga COVID-19 dado los síntomas es de {:.2f}".format(p_infectado_sintomas))
except ValueError as e:
    print(str(e))
