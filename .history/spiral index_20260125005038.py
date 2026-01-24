def spiral_index(n, m):
    """Genera una lista de indices en espiral para una matriz cuadrada de tamaño n x n."""
    indices = []
    left, right = 0, n - 1
    top, bottom = 0, n - 1

    while left <= right and top <= bottom:
        for j in range(left, right + 1):
            indices.append((top, j))
        top += 1

        for i in range(top, bottom + 1):
            indices.append((i, right))
        right -= 1

        if top <= bottom:
            for j in range(right, left - 1, -1):
                indices.append((bottom, j))
            bottom -= 1

        if left <= right:
            for i in range(bottom, top - 1, -1):
                indices.append((i, left))
            left += 1

    return indices