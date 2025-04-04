# -*- coding: utf-8 -*-
"""ae_ts_01.ipynb

Automatically generated by Colab.

Original file is located at
    
"""

# -----------------------------------------------------------------------------
# Metaheuristica Busqueda Tabú que resuelve TSP para 5 ciudades
# -----------------------------------------------------------------------------
import random

# -----------------------------------------------------------------------------
# fncion que calcula la distancia total de una solucion (recorrido)
# -----------------------------------------------------------------------------
def calcular_distancia_total(solucion, matriz_distancias):
    distancia_total = 0
    for i in range(len(solucion) - 1):
        distancia_total = distancia_total + matriz_distancias[solucion[i]][solucion[i + 1]]
    distancia_total = distancia_total + matriz_distancias[solucion[-1]][solucion[0]]  # Volver a la ciudad inicial
    return distancia_total


# -----------------------------------------------------------------------------
# funcion para generar los vecindarios intercambiando dos ciudades
# -----------------------------------------------------------------------------
def generar_vecindarios(solucion):
    vecindarios = []
    for i in range(len(solucion)):
        for j in range(i + 1, len(solucion)):
            vecindario = solucion[:]
            vecindario[i], vecindario[j] = vecindario[j], vecindario[i]  # se intercambian 2 ciudades
            vecindarios.append(vecindario)  # agrega el vecindario generado al vecindario de soluciones
    return vecindarios


# -----------------------------------------------------------------------------
# algoritmo de busqueda tabu
# -----------------------------------------------------------------------------
def busqueda_tabu(matriz_distancias, num_iteraciones, tamanio_lista_tabu):
    # inicializacion de una solucion
    solucion_actual = random.sample(range(len(matriz_distancias)), len(matriz_distancias))  # solucion inicial
    mejor_solucion = solucion_actual[:]
    print("solucion inicial: ", mejor_solucion)
    mejor_distancia = calcular_distancia_total(mejor_solucion, matriz_distancias)

    # creacion de la lista tabu
    lista_tabu = []  # la lista tabu guarda las mejores soluciones recientes

    for iteracion in range(num_iteraciones):
        # se generan los vecindarios
        vecindarios = generar_vecindarios(solucion_actual)

        # inicializacion de variables para encontrar el mejor vecindario
        mejor_vecindario = None
        mejor_vecindario_distancia = float('inf')

        # evaluacion de cada vecindario
        for vecindario in vecindarios:
            distancia_vecindario = calcular_distancia_total(vecindario, matriz_distancias)

            # aqui se verifica si la solucion esta en la lista tabu o si cumple el criterio de aspiracion
            if vecindario not in lista_tabu:
                if distancia_vecindario < mejor_vecindario_distancia:
                    mejor_vecindario = vecindario
                    mejor_vecindario_distancia = distancia_vecindario

        solucion_actual = mejor_vecindario[:] # se actualiza la solucion actual con el mejor vecindario encontrado

        lista_tabu.append(mejor_vecindario) # se agrega el mejor vecindario encontrado a la lista tabu

        # control del tamaño de la lista tabu
        if len(lista_tabu) > tamanio_lista_tabu:
            lista_tabu.pop(0)  # borro la solucion mas antigua de la lista tabu

        # se actualiza aqui la mejor solucion global (si es que encuentro una mejor)
        if mejor_vecindario_distancia < mejor_distancia:
            mejor_solucion = mejor_vecindario[:]
            mejor_distancia = mejor_vecindario_distancia

        print(f"iteracion {iteracion+1}: la mejor distancia parcial es {mejor_distancia}") # muestro por consola el progreso de la optimizacion

    return mejor_solucion, mejor_distancia


# -----------------------------------------------------------------------------
# parametros
# -----------------------------------------------------------------------------
# distanc9as entre ciudades
matriz_distancias = [
    [0, 2, 9, 10, 7],
    [2, 0, 6, 4, 3],
    [9, 6, 0, 8, 5],
    [10, 4, 8, 0, 6],
    [7, 3, 5, 6, 0]
]
num_iteraciones = 10
tamanio_lista_tabu = 5  # tambien se llama tenencia tabu (tabu tenure)


# -----------------------------------------------------------------------------
# ejecucion del algoritmo de busqueda tabu
# -----------------------------------------------------------------------------
mejor_solucion, mejor_distancia = busqueda_tabu(matriz_distancias, num_iteraciones, tamanio_lista_tabu)

print(f"\nmejor solucion encontrada: {mejor_solucion}")
print(f"la distancia total es: {mejor_distancia}")