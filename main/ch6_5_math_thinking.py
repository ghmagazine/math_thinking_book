import numpy as np

def sequential_approximation(x0, y0, z0, satellite_positions, satellite_distances, iterations=10):
    """
    与えられた衛星の座標と受信機までの距離を基にして、逐次近似法を用いて受信機の座標を推定する。
    """
    for _ in range(iterations):
        delta_x, delta_y, delta_z = 0, 0, 0  # Δx, Δy, Δzの初期化
        for sat, pos in satellite_positions.items():
            x_sat, y_sat, z_sat = pos
            distance = np.sqrt((x0 - x_sat)**2 + (y0 - y_sat)**2 + (z0 - z_sat)**2)
            delta = satellite_distances[sat] - distance
            # 勾配に基づく座標の更新量を計算
            delta_x += (x0 - x_sat) / distance * delta
            delta_y += (y0 - y_sat) / distance * delta
            delta_z += (z0 - z_sat) / distance * delta
        # 座標の更新
        x0 += delta_x
        y0 += delta_y
        z0 += delta_z
    return x0, y0, z0

# 衛星の座標と受信機までの距離
satellite_positions = {
    'A': (2200, 2000, 20500),
    'B': (1200, 2700, 20200),
    'C': (1000, 2900, 20000)
}

satellite_distances = {
    'A': 13600,
    'B': 14000,
    'C': 13800
}

# 受信機の初期座標
initial_position = (0, 0, 6400)

# 逐次近似法による受信機の座標の推定（反復回数を1000回に設定）
estimated_position = sequential_approximation(*initial_position, satellite_positions, satellite_distances, iterations=1000)

print(estimated_position)
