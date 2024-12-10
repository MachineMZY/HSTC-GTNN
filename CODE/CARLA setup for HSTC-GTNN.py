import carla
import random
import time
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, TransformerConv

# 定义HSTC-GTNN模型
class SpatialLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialLayer, self).__init__()
        self.gcn1 = GCNConv(in_channels, 128)
        self.transformer_conv = TransformerConv(128, out_channels, heads=4)

    def forward(self, x, edge_index):
        x = torch.relu(self.gcn1(x, edge_index))
        x = self.transformer_conv(x, edge_index)
        return x

class TemporalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, seq_length):
        super(TemporalLayer, self).__init__()
        self.temporal_conv = nn.Conv1d(in_channels, 128, kernel_size=3, padding=1)
        self.transformer = nn.Transformer(d_model=128, nhead=4, num_encoder_layers=2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.temporal_conv(x))
        x = x.permute(2, 0, 1)
        x = self.transformer(x)
        return x[-1]

class CrossLayerInteraction(nn.Module):
    def __init__(self, spatial_out_channels, temporal_out_channels):
        super(CrossLayerInteraction, self).__init__()
        self.fc1 = nn.Linear(spatial_out_channels + temporal_out_channels, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, spatial_output, temporal_output):
        x = torch.cat([spatial_output, temporal_output], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc_out(x)

class HSTC_GTNN(nn.Module):
    def __init__(self, node_features, temporal_seq_length, out_channels):
        super(HSTC_GTNN, self).__init__()
        self.spatial_layer = SpatialLayer(in_channels=node_features, out_channels=out_channels)
        self.temporal_layer = TemporalLayer(in_channels=node_features, out_channels=out_channels, seq_length=temporal_seq_length)
        self.cross_layer = CrossLayerInteraction(spatial_out_channels=out_channels, temporal_out_channels=out_channels)

    def forward(self, node_features, edge_index, temporal_seq):
        spatial_output = self.spatial_layer(node_features, edge_index)
        temporal_output = self.temporal_layer(temporal_seq)
        output = self.cross_layer(spatial_output, temporal_output)
        return output

# CARLA客户端与任务分配
class CarlaTaskAllocation:
    def __init__(self, model):
        # 初始化CARLA客户端
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # 加载HSTC-GTNN模型
        self.model = model

        # 获取车辆并设置控制器
        self.vehicles = []

    def spawn_vehicle(self):
        # 设置车辆生成点和蓝图
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)

        # 生成车辆
        vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicles.append(vehicle)
        print(f"Spawned vehicle {vehicle.id} at {spawn_point.location}")

    def get_vehicle_state(self):
        # 获取车辆位置和速度信息
        node_features = []
        for vehicle in self.vehicles:
            location = vehicle.get_location()
            velocity = vehicle.get_velocity()
            node_features.append([location.x, location.y, velocity.x, velocity.y])

        return torch.tensor(node_features, dtype=torch.float32)

    def allocate_task(self):
        # 模拟任务数据
        num_nodes = len(self.vehicles)
        edge_index = torch.randint(0, num_nodes, (2, 2 * num_nodes))
        temporal_seq_data = torch.rand(1, 10, 4)

        # 任务分配预测
        self.model.eval()
        with torch.no_grad():
            node_features = self.get_vehicle_state()
            output = self.model(node_features, edge_index, temporal_seq_data)
            task_allocation = output.argmax().item()
            print(f"Allocated Task ID: {task_allocation}")

    def run(self):
        try:
            while True:
                # 获取车辆状态并进行任务分配
                self.allocate_task()
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            # 清理CARLA环境中的车辆
            for vehicle in self.vehicles:
                vehicle.destroy()
            print("Cleaned up vehicles.")

if __name__ == "__main__":
    # 初始化模型
    model = HSTC_GTNN(node_features=4, temporal_seq_length=10, out_channels=64)

    # 启动CARLA任务分配系统
    carla_system = CarlaTaskAllocation(model)
    carla_system.spawn_vehicle()  # 生成测试车辆
    carla_system.run()  # 开始任务分配循环
