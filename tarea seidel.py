import numpy as np

# Definir la matriz de coeficientes y el vector de resultados
A = np.array([[52, 20, 25],
              [30, 50, 20],
              [18, 30, 55]])

B = np.array([4800, 5810, 5690])

# Definir la función para el método de Gauss-Seidel
def gauss_seidel(A, B, tol=0.005, max_iterations=1000):
    # Inicializar la primera aproximación (puede ser un vector de ceros)
    x = np.zeros_like(B, dtype=np.float64)
    
    # Iteración de Gauss-Seidel
    iter_count = 0
    while iter_count < max_iterations:
        x_new = np.copy(x)
        
        for i in range(A.shape[0]):
            suma = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (B[i] - suma) / A[i, i]
        
        # Verificar la convergencia (si la diferencia es menor que la tolerancia)
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        
        # Actualizar la solución
        x = x_new
        iter_count += 1

    return x, iter_count

# Ejecutar el método de Gauss-Seidel
solucion, iteraciones = gauss_seidel(A, B)

if solucion is not None:
    print(f"Solución en {iteraciones} iteraciones:")
    for i, valor in enumerate(solucion, 1):
        print(f"Cantidad a transportar desde la cantera {i}: {valor:.2f} metros cúbicos")
