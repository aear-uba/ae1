# .................................................................
# Ejemplo de PSO con restricciones
#      maximizar     3x1 + 5x2
#
#      sujeto a:
#                     x1        <= 4
#                           2x2 <= 12
#                     3x1 + 2x2 <= 18
# .................................................................
import numpy as np


# función objetivo a maximizar
def f(x):
    return 3 * x[0] + 5 * x[1]  # funcion objetivo: 3x1 + 5x2


# primera restriccion
def g1(x):
    return x[0] - 4 <= 0  # restriccion: x1 <= 4


# segunda restriccion
def g2(x):
    return 2 * x[1] - 12 <= 0  # restriccion: 2x2 <= 12


# tercera restriccion
def g3(x):
    return 3 * x[0] + 2 * x[1] - 18 <= 0  # restriccion: 3x1 + 2x2 <= 18


# parametros
n_particles = 20  # numero de particulas en el enjambre
n_dimensions = 2  # dimensiones del espacio de busqueda (x1 y x2)
max_iterations = 100  # numero máximo de iteraciones para la optimizacion
c1 = c2 = 2  # coeficientes de aceleracion
w = 0.5  # Factor de inercia

# inicialización de particulas
x = np.zeros((n_particles, n_dimensions))  # matriz para las posiciones de las particulas
v = np.zeros((n_particles, n_dimensions))  # matriz para las velocidades de las particulas
pbest = np.zeros((n_particles, n_dimensions))  # matriz para los mejores valores personales
pbest_fit = -np.inf * np.ones(n_particles)  # mector para las mejores aptitudes personales (inicialmente -infinito)
gbest = np.zeros(n_dimensions)  # mejor solución global
gbest_fit = -np.inf  # mejor aptitud global (inicialmente -infinito)

# inicializacion de particulas factibles
for i in range(n_particles):
    while True:  # bucle para asegurar que la particula sea factible
        x[i] = np.random.uniform(0, 10, n_dimensions)  # inicializacion posicion aleatoria en el rango [0, 10]
        if g1(x[i]) and g2(x[i]) and g3(x[i]):  # se comprueba si la posicion cumple las restricciones
            break  # Salir del bucle si es factible
    v[i] = np.random.uniform(-1, 1, n_dimensions)  # inicializar velocidad aleatoria
    pbest[i] = x[i].copy()  # ee establece el mejor valor personal inicial como la posicion actual
    fit = f(x[i])  # calculo la aptitud de la posicion inicial
    if fit > pbest_fit[i]:  # si la aptitud es mejor que la mejor conocida
        pbest_fit[i] = fit  # se actualiza el mejor valor personal

# Optimizacion
for _ in range(max_iterations):  # Repetir hasta el número máximo de iteraciones
    for i in range(n_particles):
        fit = f(x[i])  # Se calcula la aptitud de la posicion actual
        # Se comprueba si la nueva aptitud es mejor y si cumple las restricciones
        if fit > pbest_fit[i] and g1(x[i]) and g2(x[i]) and g3(x[i]):
            pbest_fit[i] = fit  # Se actualiza la mejor aptitud personal
            pbest[i] = x[i].copy()  # Se actualizar la mejor posicion personal
            if fit > gbest_fit:  # Si la nueva aptitud es mejor que la mejor global
                gbest_fit = fit  # Se actualizar la mejor aptitud global
                gbest = x[i].copy()  # Se actualizar la mejor posicion global

        # actualizacion de la velocidad de la particula
        v[i] = w * v[i] + c1 * np.random.rand() * (pbest[i] - x[i]) + c2 * np.random.rand() * (gbest - x[i])
        x[i] += v[i]  # Se actualiza la posicion de la particula

        # se asegura de que la nueva posicion esté dentro de las restricciones
        if not (g1(x[i]) and g2(x[i]) and g3(x[i])):
            # Si la nueva posicion no es válida, revertir a la mejor posicion personal
            x[i] = pbest[i].copy()

# Se imprime la mejor solucion encontrada y también su valor optimo
print(f"Mejor solucion: [{gbest[0]:.4f}, {gbest[1]:.4f}]")
print(f"Valor optimo: {gbest_fit}")
