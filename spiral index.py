def spiral_index(n, m):
    """Genera una lista de indices de una matriz n x m en orden espiral."""
    result = []
    top, bottom, left, right = 0, n - 1, 0, m - 1

    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            result.append((top, j))
        top += 1

        for i in range(top, bottom + 1):
            result.append((i, right))
        right -= 1

        if top <= bottom:
            for j in range(right, left - 1, -1):
                result.append((bottom, j))
            bottom -= 1

        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append((i, left))
            left += 1

    return result

print(spiral_index(5, 5))  # Ejemplo de uso