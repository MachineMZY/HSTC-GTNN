import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import torch
from torch_geometric.nn import GCNConv, TransformerConv

# 定义ROS节点并连接Gazebo
class TaskAllocationNode:
    def __init__(self, model):
        # 初始化ROS节点
        rospy.init_node('task_allocation_node', anonymous=True)

        # ROS发布器和订阅器
        self.task_pub = rospy.Publisher('/task_allocation', String, queue_size=10)
        self.ugv_sub = rospy.Subscriber('/ugv/pose', Pose, self.ugv_callback)
        self.uav_sub = rospy.Subscriber('/uav/pose', Pose, self.uav_callback)

        # 模型加载
        self.model = model

    def ugv_callback(self, data):
        # 处理UGV位置数据
        self.ugv_position = data.position

    def uav_callback(self, data):
        # 处理UAV位置数据
        self.uav_position = data.position

    def predict_and_publish(self):
        # 定义数据输入（模拟或真实数据）
        num_nodes = 10
        node_features = torch.rand(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, 20))
        temporal_seq_data = torch.rand(1, 10, 64)

        # 通过模型预测任务分配
        self.model.eval()
        with torch.no_grad():
            output = self.model(node_features, edge_index, temporal_seq_data)
            task_allocation = output.argmax().item()

        # 将任务分配发布到ROS
        rospy.loginfo(f"Task Allocation: {task_allocation}")
        self.task_pub.publish(f"Allocated Task ID: {task_allocation}")

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

# 主程序入口
if __name__ == '__main__':
    try:
        # 加载HSTC-GTNN模型
        model = HSTC_GTNN(node_features=64, temporal_seq_length=10, out_channels=64)
        
        # 启动任务分配节点
        node = TaskAllocationNode(model)
        rate = rospy.Rate(1)  # 控制频率

        # 循环执行预测和任务分配发布
        while not rospy.is_shutdown():
            node.predict_and_publish()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
