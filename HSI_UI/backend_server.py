import socket
import json
import time
import random

HOST = '127.0.0.1'
PORT = 9999


def create_test_data():
    # 生成随机旋转角度（pitch, yaw, roll）
    def random_rotation():
        return {
            "pitch": round(random.uniform(-10.0, 10.0), 2),
            "yaw": round(random.uniform(-180.0, 180.0), 2),
            "roll": round(random.uniform(-180.0, 180.0), 2)
        }

    # 生成随机命令信息
    def random_command():
        return [random.randint(0, 1), random.randint(-1, 5)]  # 第一个值为命令类型（0或1），第二个值为目标编号（-1表示无目标）

    # 定义敌方组及其威胁值
    enemy_groups = [
        {"group_id": 0, "threat_level": round(random.uniform(0.5, 1.0), 2)},
        {"group_id": 1, "threat_level": round(random.uniform(0.3, 0.7), 2)},
        {"group_id": 2, "threat_level": round(random.uniform(0.1, 0.5), 2)}
    ]

    data = {
        "type": "update_positions",
        "ally": [
            {
                "id": "ally1",
                "x": 0,
                "y": 2000,
                "z": 4000,
                "vx": -127.71629333496094,
                "vy": 17.170900344848633,
                "vz": 76.77044677734375,
                "rotation": random_rotation(),
                # "command": random_command(),  # 添加命令信息
                "health_level": round(random.uniform(0.5, 1.0), 2)  # 添加健康值
            },
            {
                "id": "ally2",
                "x": 1000,
                "y": 2100,
                "z": 4500,
                "vx": -100.0,
                "vy": 20.0,
                "vz": 80.0,
                "rotation": random_rotation(),
                # "command": random_command(),
                "health_level": round(random.uniform(0.5, 1.0), 2)
            }
        ],
        "enemy": [
            {
                "id": "enemy1",
                # "group": 0,
                "x": 0,
                "y": 2000,
                "z": 0,
                "vx": -35.005958557128906,
                "vy": 0.0,
                "vz": -145.8580780029297,
                "rotation": random_rotation(),
                "health_level": round(random.uniform(0.5, 1.0), 2)  # 添加健康值
            },
            {
                "id": "enemy2",
                # "group": 1,
                "x": -1500,
                "y": 1950,
                "z": 3000,
                "vx": -50.0,
                "vy": 5.0,
                "vz": -120.0,
                "rotation": random_rotation(),
                "health_level": round(random.uniform(0.5, 1.0), 2)
            },
            {
                "id": "enemy3",
                # "group": 2,
                "x": 2000,
                "y": 2050,
                "z": -2500,
                "vx": -20.0,
                "vy": 10.0,
                "vz": -100.0,
                "rotation": random_rotation(),
                "health_level": round(random.uniform(0.5, 1.0), 2)
            }
        ],
        # "enemy_groups": enemy_groups,  # 添加敌方组及其威胁值
        "agent_preference": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        "mixed_priority": [[1, 2, 3, 4, 5, 6]]
    }
    return data


def run_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 允许重用地址
        s.bind((HOST, PORT))
        s.listen()
        print(f"后端服务器正在运行在 {HOST}:{PORT}")

        while True:
            try:
                conn, addr = s.accept()
                with conn:
                    print(f"连接来自 {addr}")
                    while True:
                        data = create_test_data()
                        msg = json.dumps(data) + "\n"
                        try:
                            conn.sendall(msg.encode('utf-8'))
                            print("发送数据:", msg.strip())
                        except (BrokenPipeError, ConnectionResetError):
                            print("客户端已断开连接")
                            break
                        time.sleep(5)  # 每5秒发送一次数据
            except KeyboardInterrupt:
                print("\n服务器已手动停止。")
                break
            except Exception as e:
                print(f"发生异常: {e}")
                break


if __name__ == "__main__":
    run_server()
