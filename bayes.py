# Definir las probabilidades condicionales de las porpabilidades de covid
p_fatiga = 0.75
p_tos_seca =  0.80
p_dificultad_respirar = 0.90
p_dolor_garganta = 0.65
p_dolor_cabeza = 0.20
p_dolor_cuerpo = 0.30
p_escalofrios = 0.40
p_secresion_nasal = 0.50
p_perdida_sentido = 0.70
p_fiebre = 0.80
p_dolor_pecho = 0.70

# Definir la prevalencia dle covid en la población
p_infectado = 0.05

#Definir los sintomas obsrevados en el paciente
sintomas = ['Fatiga', 'Tos seca', 'Dificuktad para respirar', 'Dolor de garganta','Dolor de cabeza', 
            'Dolor en el cuerpo', 'Escalofrios', 'Secresión nasal','Pérdida sentido del olfato', 'Fiebre',
              'Dolor de pecho']

# Calcular la probabilidad de que el paciente tenga covid 19
p_sintomas_infectado = p_fatiga * p_tos_seca * p_dificultad_respirar * p_dolor_garganta * p_dolor_cabeza * p_dolor_cuerpo * \
p_escalofrios * p_secresion_nasal * p_perdida_sentido * p_fiebre * p_dolor_pecho

# Se asume que la probabilidad de tener covid 19 sin tener los sintomas es muy baja
p_sintomas_no_infectado = 0.05 ** 11

p_sintomas = p_sintomas_infectado * p_infectado + p_sintomas_no_infectado * (1-p_infectado)

p_infectado_sintomas = p_sintomas_infectado * p_infectado / p_sintomas

print(f"La probabilidad que el paciente tenga covid dado los sintomas es de {p_infectado_sintomas}")
