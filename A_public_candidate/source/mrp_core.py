"""Truth-table construction, MRP placement, and neutral export core.

Definitions are extracted unchanged from MNN_to_matrix_v7.py. The old
training, plotting, and simulator entry points are intentionally excluded.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from binarized_modules import BinarizeLinear, Linears
from DeepBMNN_modules1_v5 import Out_0_or_1


def save_layer_params_to_dict(module, layer_prefixes=None, indent=0):
    if layer_prefixes is None:
        layer_prefixes = {}
    for name, sub_module in module.named_children():
        index = ''.join([char for char in name if char.isdigit()])
        key = f'MLUTy{index}'
        if key not in layer_prefixes:
            layer_prefixes[key] = {
                'weight': None,
                'bias': None,
                'bn_weight': None,
                'bn_bias': None,
                'running_mean': None,
                'running_var': None,
                'activation': None,
                'B_0_or_1': False,
            }

        # --- Linear / BinarizeLinear / Linears の重み・バイアス ---
        if isinstance(sub_module, (BinarizeLinear, Linears, nn.Linear)):
            if hasattr(sub_module, 'weight') and sub_module.weight is not None:
                layer_prefixes[key]['weight'] = sub_module.weight.detach().clone()
            if hasattr(sub_module, 'bias') and sub_module.bias is not None:
                layer_prefixes[key]['bias'] = sub_module.bias.detach().clone()

        if isinstance(sub_module, nn.BatchNorm1d):
            if hasattr(sub_module, 'weight') and sub_module.weight is not None:
                layer_prefixes[key]['bn_weight'] = sub_module.weight.detach().clone()
            if hasattr(sub_module, 'bias') and sub_module.bias is not None:
                layer_prefixes[key]['bn_bias'] = sub_module.bias.detach().clone()
            if hasattr(sub_module, 'running_mean') and sub_module.running_mean is not None:
                layer_prefixes[key]['running_mean'] = sub_module.running_mean.detach().clone()
            if hasattr(sub_module, 'running_var') and sub_module.running_var is not None:
                layer_prefixes[key]['running_var'] = sub_module.running_var.detach().clone()

        if isinstance(sub_module, nn.Hardtanh):
            layer_prefixes[key]['activation'] = 'Hardtanh'
        elif isinstance(sub_module, nn.LogSoftmax):
            layer_prefixes[key]['activation'] = 'LogSoftmax'
        elif isinstance(sub_module, nn.SiLU):
            layer_prefixes[key]['activation'] = 'SiLU'

        if isinstance(sub_module, Out_0_or_1):
            layer_prefixes[key]['B_0_or_1'] = True

        if len(list(sub_module.named_children())) != 0:
            save_layer_params_to_dict(sub_module, layer_prefixes, indent + 2)

    return layer_prefixes


def save_layer_module_to_dict(module ,layer_prefixes=None, indent=0):
    if layer_prefixes is None:
        layer_prefixes = {}
    for name, sub_module in module.named_children():
        index = ''.join([char for char in name if char.isdigit()])
        key = f'MLUTy{index}'
        if key not in layer_prefixes:
            layer_prefixes[key] = []
        if sub_module is not None:
            layer_prefixes[key].append(sub_module)
        
        if len(list(sub_module.named_children())) != 0:
            save_layer_params_to_dict(sub_module, layer_prefixes, indent + 2)
    return layer_prefixes


def adjust_node_positions(G, max_nodes_per_layer):
    pos = nx.get_node_attributes(G, 'pos')
    
    for layer in range(max_nodes_per_layer):
        nodes_in_layer = [node for node, attr in G.nodes(data=True) if attr['layer'] == layer]
        
        if len(nodes_in_layer) == 0:
            continue
        
        offset = (max_nodes_per_layer - len(nodes_in_layer)) // 2
        
        for i, node in enumerate(nodes_in_layer):
            x, y = pos[node]
            if layer % 2 == 1:  # 奇数番目の層
                y += offset
            else:  # 偶数番目の層
                y -= offset
            pos[node] = (x, y)
    
    nx.set_node_attributes(G, pos, 'pos')
    return G


def normalize_flow_direction(Flip=True, flow_direction=None):
    if flow_direction is None:
        return "fold" if Flip else "left_to_right"
    flow_direction = str(flow_direction).lower().replace("-", "_")
    aliases = {
        "lr": "left_to_right",
        "left": "left_to_right",
        "left_to_right": "left_to_right",
        "ltr": "left_to_right",
        "right": "right_to_left",
        "right_to_left": "right_to_left",
        "rtl": "right_to_left",
        "fold": "fold",
        "folded": "fold",
    }
    if flow_direction not in aliases:
        raise ValueError(f"Unsupported flow_direction: {flow_direction}")
    return aliases[flow_direction]


def uses_right_to_left_ports(num_layer_target, num_layers, Flip=True, flow_direction=None):
    flow_direction = normalize_flow_direction(Flip, flow_direction)
    if flow_direction == "right_to_left":
        return True
    if flow_direction == "left_to_right":
        return False
    return num_layer_target >= (num_layers - 1) / 2


def default_side_ports(width, max_width, pos_y, side):
    if side == "left":
        if width == max_width:
            return list(range(0, width))
        if pos_y == 0:
            return list(range(width, width * 2))
        return list(range(0, width))
    if side == "right":
        if width == max_width:
            return list(reversed(range(width, width * 2)))
        if pos_y == 0:
            return list(reversed(range(width * 2, width * 3)))
        return list(reversed(range(width * 3, width * 4)))
    raise ValueError(f"Unsupported side: {side}")


def physical_x_for_flow(layer_index, layer_num, flow_direction):
    if flow_direction == "right_to_left":
        return layer_num - 1 - layer_index
    if flow_direction == "fold":
        center = (layer_num - 1) // 2
        return center - abs(center - layer_index)
    return layer_index


def column_shape_from_flip(physical_x, flip_up):
    return "up" if (flip_up + physical_x) % 2 == 1 else "down"


def next_layer_top_y(prev_top_y, prev_count, current_count, prev_x, current_x, flip_up):
    diff = abs(current_count - prev_count)
    if diff == 0:
        return prev_top_y

    prev_shape = column_shape_from_flip(prev_x, flip_up)
    current_shape = column_shape_from_flip(current_x, flip_up)

    if current_count < prev_count:
        if prev_shape == "down" and current_shape == "up":
            return prev_top_y + diff
        if prev_shape == "up" and current_shape == "down":
            return prev_top_y
        return prev_top_y + diff if current_shape == "up" else prev_top_y

    if prev_shape == "up" and current_shape == "down":
        return prev_top_y - diff
    if prev_shape == "down" and current_shape == "up":
        return prev_top_y
    return prev_top_y - diff if current_shape == "down" else prev_top_y


def compute_layer_top_y(layer_counts, layer_num, flip_up, flow_direction):
    if not layer_counts:
        return []

    top_y_by_layer = [0]
    for i in range(1, len(layer_counts)):
        prev_x = physical_x_for_flow(i - 1, layer_num, flow_direction)
        current_x = physical_x_for_flow(i, layer_num, flow_direction)
        top_y_by_layer.append(
            next_layer_top_y(
                top_y_by_layer[-1],
                layer_counts[i - 1],
                layer_counts[i],
                prev_x,
                current_x,
                flip_up,
            )
        )
    return top_y_by_layer


def determination_io_port(pos1,pos2,num_layer_target,num_layers,input_size,output_size,Flip,flow_direction=None):
    #print(num_layer_target,num_layers/2)
    #print()
    if not uses_right_to_left_ports(num_layer_target, num_layers, Flip, flow_direction):#if before switch layer
        if pos1[1]>pos2[1]: #if node2 above node1
            input_port=list((range(0,input_size//2)))
            output_port=list(reversed(range(output_size,output_size+output_size//2)))
        else:
            input_port=list((range(input_size//2,input_size)))
            output_port=list(reversed(range(output_size+output_size//2,output_size*2)))
        
    else:                               #if switch layer and after
        if pos1[1]>pos2[1]: #if node2 above node1
            input_port=list(reversed(range(input_size+input_size//2,input_size*2)))
            output_port=list((range(output_size//2,output_size)))
        else:
            input_port=list(reversed(range(input_size,input_size+input_size//2)))
            output_port=list((range(0,output_size//2)))
            
    
    return input_port,output_port#input_port:input port of node2,output_port:output port of node1


def find_max_element_in_network_params(G, key):
    if key in G.graph['input_output_types']:
        max_value = max(G.graph['input_output_types'][key])
        return max_value
    else:
        return None


def add_unique_element_to_network_params(G, key, value):
    if key in G.graph['input_output_types']:
        if value not in G.graph['input_output_types'][key]:
            G.graph['input_output_types'][key].append(value)
       
    else:
        print(f"Key {key} not found in the network parameters")


def Binarize(tensor):    # -> -1 or 1
    return torch.where(tensor <= 0, -1, 1).float()


def binarized_linear(input, weight, bias, binary_t_f=True):
    if binary_t_f==True:
        input = Binarize(input)
    out = F.linear(input, weight, bias)
    return out


def apply_batch_norm(x, bn_weight, bn_bias, mean=None, var=None):
    if mean is None or var is None:
        mean = x.mean(0)
        var = x.var(0, unbiased=False)
    x_normalized = (torch.tensor(x) - torch.tensor(mean)) / torch.sqrt(torch.tensor(var) + 1e-5)
    return torch.tensor(bn_weight) * torch.tensor(x_normalized) + torch.tensor(bn_bias)


def apply_activation(x, activation):  # 活性化関数の適用
    if activation == 'Hardtanh':
        return torch.clamp(x, min=-1, max=1)
    elif activation == 'LogSoftmax':
        return F.log_softmax(x, dim=-1)
    else:
        raise NotImplementedError(f"Activation function '{activation}' is not implemented.")


def apply_binarization(x, flag_0_1=True, device='cpu'):  # deviceを追加
    if flag_0_1:
        return torch.where(x <= 0, torch.tensor(0, dtype=torch.float, device=device), torch.tensor(1, dtype=torch.float, device=device))
    else:
        return x


def _module_device(module):
    if isinstance(module, nn.Module):
        for tensor in module.parameters(recurse=True):
            return tensor.device
        for tensor in module.buffers(recurse=True):
            return tensor.device
    return None


def _layers_device(layers, fallback='cpu'):
    for layer in layers:
        device = _module_device(layer)
        if device is not None:
            return device
    return torch.device(fallback)


def int_to_binary_list(num, n):
    # 二進数の文字列に変換
    binary_str = format(num, f'0{n}b')
    # 文字列をリストに変換
    binary_list = [int(bit) for bit in binary_str]
    return binary_list


def analyze_layer_dependencies(layers):
    layers_have_2d_weight=[[0, 0] for _ in range(len(layers))]
    ##layers_have_2d_weight[index][0]=0:input[n]->output[n],1:input[0~m]->output[n],2:input[k~l]->output[s]
    index=0
    for layer in layers:
        if hasattr(layer, 'weight'):
            # 重みが存在し、2次元かどうかを確認
            if layer.weight.dim() == 2:
                layers_have_2d_weight[index][0]=1       
                layers_have_2d_weight[index][1]=layer.weight.detach().clone()
                #print("add_weight")
        else:
            # 重みがない場合、平均プーリングなどかを確認
            if isinstance(layer, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
                layers_have_2d_weight[index][0]=2
                layers_have_2d_weight[index][1]=layer.kernel_size
            else:
                layers_have_2d_weight[index][0]=0
                layers_have_2d_weight[index][1]=0
        index+=1
    #print("###",layers_have_2d_weight,"\n\n")
    return layers_have_2d_weight


def visualize_ml_net_comp(input_dim, output_dim, layers, iosize_dp):
    # グラフの初期化
    G = nx.DiGraph()
    layers_io_relation=analyze_layer_dependencies(layers)
    # ノードの追加（入力ノード）
    input_nodes = [f'input_{i}' for i in range(input_dim)]                  #MLuTの入力部分
    hidden_layer_num=len(layers)
    
   
    # ノードの追加（隠れノード）
    hidden_nodes=[[] for _ in range(hidden_layer_num)]
    for j in range(hidden_layer_num):
        hidden_nodes[j]=[f'hidden_{j}_{i}' for i in range(iosize_dp[j]["output_size"])]      
        
    # ノードの追加（出力ノード）
    output_nodes= [f'output_{i}' for i in range(output_dim)]
    
    #すべてのノードの管理用配列
    all_nodes=[]
    all_nodes.append(input_nodes)
    for n in range(len(hidden_nodes)):
        all_nodes.append(hidden_nodes[n])
    all_nodes.append(output_nodes)
    
    
    #ノードのネットワークへの追加
    G.add_nodes_from(input_nodes, layer='input')
    for i in range(hidden_layer_num):
        G.add_nodes_from(hidden_nodes[i], layer='hidden')
    G.add_nodes_from(output_nodes, layer='output')
    #エッジの追加
    for i in range(hidden_layer_num):
        if layers_io_relation[i][0]==1:#この層は線形層である
            for node1 in all_nodes[i+1]:
                for node2 in all_nodes[i]:
                        G.add_edge(node2, node1)
        elif layers_io_relation[i][0]==0:#この層は入力と出力の関係が1対1である
            for j in range(iosize_dp[i]["input_size"]):
                G.add_edge(all_nodes[i][j], all_nodes[i+1][j])
                    
        elif layers_io_relation[i][0]==2:#この層はプーリング層である
            for j in iosize_dp[i]["input_size"]:
                for k in  layers_io_relation[i][1]:
                            G.add_edge(all_nodes[i][j], all_nodes[i+1][j+k])
        else:
            continue
    # print(all_nodes,"\n################\n\n",i)
    for j in range(output_dim):
        # print("j=",j)
        G.add_edge(all_nodes[i+1][j], all_nodes[i+2][j])
    
    return G


def get_1d_io_sizes(layers, max_probe_size=4096):
    device = _layers_device(layers)
    last_error = None
    for size in range(1, max_probe_size + 1):
        try:
            output = torch.randn(1, size, device=device)
            with torch.no_grad():
                for layer in layers:
                    output = layer(output)
            return {'input_size': size, 'output_size': output.size(1)}
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Could not infer 1D IO size up to {max_probe_size} on {device}: "
        f"{last_error}"
    )


def get_1d_io_size(layer, size_min=1, max_probe_size=4096):
    device = _module_device(layer) or torch.device('cpu')
    last_error = None
    for size in range(size_min, max_probe_size + 1):
        try:
            dummy_input = torch.randn(1, size, device=device)
            with torch.no_grad():
                output = layer(dummy_input)
            return {'input_size': size, 'output_size': output.size(1)}
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Could not infer 1D IO size for {type(layer).__name__} "
        f"up to {max_probe_size} on {device}: {last_error}"
    )


def add_edges_based_on_positions_comp(G,N_of_Layers=0,Flip=True,flow_direction=None):
    flow_direction = normalize_flow_direction(Flip, flow_direction)
    positions = nx.get_node_attributes(G, 'pos')
    for node1, pos1 in positions.items():
        input_port_all=[]
        output_port_all=[]
        # print("***",node1,pos1)
        pos_y=G.nodes[node1]['pos_y']
        #green input blue output
        for node2, pos2 in positions.items():
            input_port,output_port=[],[]
            if (pos1[0] == pos2[0] - 1) and abs(pos1[1]-pos2[1])<=1:  # 隣接する層のノード
                input_port,output_port=determination_io_port(pos1, pos2, pos1[0], N_of_Layers, find_max_element_in_network_params(G, 'input_counts'),find_max_element_in_network_params(G, 'output_counts'),Flip,flow_direction)
                # print(node1,node2,pos1,pos2)
                G.add_edge(node1, node2, input_port=input_port,output_port=output_port)
                output_port_all.append(output_port) 
                # print(node2,pos2)
                # print(output_port)
            if (pos1[0] == pos2[0] + 1) and abs(pos1[1]-pos2[1])<=1:  # 隣接する層のノード
                input_port,output_port=determination_io_port(pos2, pos1, pos2[0], N_of_Layers, find_max_element_in_network_params(G, 'input_counts'),find_max_element_in_network_params(G, 'output_counts'),Flip,flow_direction)
                
                input_port_all.append(input_port)
                
        if len(input_port_all)==0:
            input_side = "right" if flow_direction == "right_to_left" else "left"
            input_port_all = default_side_ports(
                G.nodes[node1]['io_in'],
                find_max_element_in_network_params(G, 'input_counts'),
                pos_y,
                input_side,
            )
            
        if len(output_port_all)==0:
            if flow_direction in ("fold", "right_to_left"):
                 if G.nodes[node1]['io_out']==find_max_element_in_network_params(G, 'output_counts'):
                     output_port_all=list((range(0,G.nodes[node1]['io_out'])))
                 elif pos_y==0:
                     output_port_all=list((range(G.nodes[node1]['io_out'],G.nodes[node1]['io_out']*2)))
                 
                 else:
                     output_port_all=list((range(0,G.nodes[node1]['io_out']))) 
                
            else:
                # print(G.nodes[node1]['io_out'])
                if G.nodes[node1]['io_out']==find_max_element_in_network_params(G, 'output_counts'):
                    output_port_all=list((range(G.nodes[node1]['io_out'],G.nodes[node1]['io_out']*2)))
                elif pos_y==0:
                    output_port_all=list((range(G.nodes[node1]['io_out']*2,G.nodes[node1]['io_out']*2+G.nodes[node1]['io_out'])))
                
                else:
                    output_port_all=list((range(G.nodes[node1]['io_out']*2+G.nodes[node1]['io_out'],G.nodes[node1]['io_out']*4))) 
        
        G.nodes[node1]['output_port']=sum(output_port_all, []) if len(output_port_all)!=G.nodes[node1]['io_out'] else output_port_all
        
        G.nodes[node1]['input_port']=sum(input_port_all, [])  if len(input_port_all)!=G.nodes[node1]['io_in'] else input_port_all
        #print(G.nodes[node1]['input_port'],G.nodes[node1]['output_port'],"####",input_port_all,node1,G.nodes[node1]['io_out'],G.nodes[node1]['io_in'])
    return G


def visualize_ml_ml_net_advanse(collected_params,Flip=True,flow_direction=None):
    
    flow_direction = normalize_flow_direction(Flip, flow_direction)
    G = nx.DiGraph()
    positions = {}
    max_x=0
    current_x = 0
    flip_up=0
    max_layer_node=0
    input_output_types = {
        'input_counts': [],  # 各層の入力数のリスト
        'output_counts': []  # 各層の出力数のリスト
    }
    max_layer_items=0
    node_name_first=None
    G.graph['input_output_types'] = input_output_types
    #入出力サイズの決定
    for i in range(len(collected_params)):
        layer_params=collected_params[i]
        for j ,params in layer_params.items():
            io_size=get_1d_io_sizes(params)
            add_unique_element_to_network_params(G, 'input_counts', io_size['input_size'])
            add_unique_element_to_network_params(G, 'output_counts', io_size['output_size'])

    layer_num=len(collected_params.values())
    if layer_num > 0:
        first_layer_params = collected_params[0]
        if len(first_layer_params) > 0:
            first_params = next(iter(first_layer_params.values()))
            first_io_size = get_1d_io_sizes(first_params)
            if first_io_size['input_size'] == min(G.graph['input_output_types']['input_counts']):
                flip_up = 1

    layer_counts = [len(collected_params[i]) for i in range(layer_num)]
    layer_top_y = compute_layer_top_y(layer_counts, layer_num, flip_up, flow_direction)

    for i in range(len(collected_params)):
        layer_params=collected_params[i]
        if len(layer_params)>max_layer_node:
            max_layer_node=len(layer_params)

        current_y_MRLD = layer_top_y[i]
        current_y = -current_y_MRLD
        l=0
        pos_x_MRLD = physical_x_for_flow(i, layer_num, flow_direction)
        for j ,params in layer_params.items():
            node_name = f'{i}_{j}'
            if node_name_first==None:
                node_name_first=node_name
            io_size=get_1d_io_sizes(params)
            io_size_dp=[]
            input_size_min=1
            for k in range(len(params)):
                io_size_dp.append(get_1d_io_size(params[k],input_size_min))
                input_size_min=io_size_dp[k]["output_size"]
            #print(io_size,params)
            
            if l==0 and (flip_up+pos_x_MRLD)%2==0:
                current_y-=0.5

            G.add_node(node_name, layer='main', pos_x=i , pos_y=l , pos_x_MRLD=pos_x_MRLD, pos_y_MRLD=l+current_y_MRLD  ,layers=params, io_in=io_size['input_size'],io_out=io_size['output_size'],io_per_layer=io_size_dp)
            #print("$$4",min(G.graph['input_output_types']['input_counts']),G.graph['input_output_types']['input_counts'])

            sub_net = visualize_ml_net_comp(io_size['input_size'], io_size['output_size'], params, io_size_dp)
            #input_dim, output_dim, layers, iosize_dp
            G.nodes[node_name]['subnet'] = sub_net
            positions[node_name] = (current_x, current_y)
            # print(G.nodes[node_name]['pos_x'],G.nodes[node_name]['pos_y'],G.nodes[node_name]['pos_x_MRLD'],G.nodes[node_name]['pos_y_MRLD'],)
            current_y -= 1   
            l+=1
            if l > max_layer_items :
                max_layer_items=l
            if max_x<G.nodes[node_name]['pos_x_MRLD']+1:
                max_x=G.nodes[node_name]['pos_x_MRLD']+1
        current_x += 1
        max_nodes_per_layer = max(len(layer) for layer in collected_params.values())
    nx.set_node_attributes(G, positions, 'pos')
    # ネットワーク全体の属性として配列を追加
    G = add_edges_based_on_positions_comp(G,layer_num,Flip,flow_direction)      #エッジの追加
    G = adjust_node_positions(G, max_nodes_per_layer)       #ノードの位置の追加
    G.graph["max_layer_items"]=max_layer_items
    G.graph["flip_up"]=flip_up
    G.graph["Flip"]=Flip
    G.graph["flow_direction"]=flow_direction
    G.graph["layer_num"]=current_x
    G.graph["max_x"]=max_x
    return G


def Create_Truth_table_comp(G,nodename,device='cpu'):#8*4,8*8,4*4
    Table=[]
    input_port=G.nodes[nodename]['input_port']
    layers = G.nodes[nodename]["layers"]
    device = _layers_device(layers, fallback=device)
    
    # print(G.nodes[nodename]['weight'],"\n $$", G.nodes[nodename]['bias'])
    for i in range(0,pow(2,len(input_port))):
        binary_list = int_to_binary_list(i, len(input_port))

        # Map 0 -> -1, 1 -> +1
        mapped_list = [2 * b - 1 for b in binary_list]
        
        # Create input tensor of shape (1, n)
        input_tensor = torch.tensor(mapped_list, dtype=torch.float, device=device).unsqueeze(0)
        out=input_tensor.to(device)
        for layer in layers:
            out=layer(out.to(device))
            
        out = apply_binarization(out, True, device=device)
        
        Table.append(out)
    return Table


def Create_Truth_table_MRLD_comp(G, nodename, device='cpu'):  # 8*16, 8*16
    Truth_table_base = Create_Truth_table_comp(G, nodename, device)

    Truth_table = []
    input_port = G.nodes[nodename]['input_port']
    output_port = G.nodes[nodename]['output_port']
    max_input = find_max_element_in_network_params(G, 'input_counts')
    max_output = find_max_element_in_network_params(G, 'output_counts')
    pos_x_MRLD = G.nodes[nodename]['pos_x_MRLD']
    pos_x = G.nodes[nodename]['pos_x']
    end_index_i = input_port[-1]
    start_index_o = output_port[0]
    end_index_o = output_port[-1]
    #print(nodename,input_port)
    #print(G.nodes[nodename]["layers"],"\n")
    for i in range(0, 2**max_input):
        insert_MLUT = torch.zeros(max_output * 2, dtype=torch.int)
        if len(input_port)==max_input:
            # Vivado forms both SRAM addresses in packed-vector order.
            j=i

        elif abs(end_index_i-(max_input-0.5)) <  max_input // 2:
            j = i % 2**(max_input // 2)
        else:
            j = i // 2**(max_input // 2)
            
        
        base_slice = torch.as_tensor(
            Truth_table_base[j], dtype=torch.int, device=insert_MLUT.device
        )
        # if i<10:
        #     print(base_slice,"#########")
        #     print(start_index_o,end_index_o)
        # print((base_slice[len(base_slice)//2:],base_slice[:len(base_slice)//2]))
        #出力の成形部
        if start_index_o > end_index_o:  # submit output
            end_index_o_r = start_index_o
            start_index_o_r = end_index_o
            insert_MLUT[start_index_o_r:end_index_o_r+1] =torch.flip(base_slice.squeeze(),dims=[0])

        else:
            
            insert_MLUT[start_index_o:end_index_o+1] = base_slice
            # if i<5 or i>250:
            #         print(i,j,"::::",nodename)
            #         print(base_slice,"::::",insert_MLUT)
            
        # if i<10:        
        #     print(insert_MLUT)
        Truth_table.append(insert_MLUT)
        
        # if pos_x==5 and i==23:
        #         print(torch.tensor(Truth_table_base[i], dtype=torch.int),i)
        #         print(torch.tensor(Truth_table_base[j], dtype=torch.int),j)
        #         print(base_slice)
        #         print(insert_MLUT)
    try:
        print(start_index_o_r ,end_index_o_r)
    except:
        print(start_index_o ,end_index_o)
    return torch.stack(Truth_table)


def insert_Truth_table_comp(G):
    i=0
    for nodename in G.nodes:
        i+=1
        G.nodes[nodename]['Truth_Table']=Create_Truth_table_comp(G,nodename)
        G.nodes[nodename]['Truth_Table_MRLD']=Create_Truth_table_MRLD_comp(G,nodename)

            
    return G


def input_layer(input_data,params):
    # print(params)
    weight=params['weight']
    bias=params['bias']
    bn_weight=params['bn_weight']
    bn_bias=params['bn_bias']
    activation=params['activation']
    mean=params['running_mean']
    var=params['running_var']
    out=binarized_linear(input_data,weight, bias, binary_t_f=False)
    out=apply_batch_norm(out, bn_weight, bn_bias,mean=mean,var=var)
    out=apply_activation(out, activation)
    out=apply_binarization(out, True)
    return out.clone().detach()


def output_layer(input_data,params):
    
    weight=params['weight']
    bias=params['bias']
    bn_weight=params['bn_weight']
    bn_bias=params['bn_bias']
    activation=params['activation']
    mean=params['running_mean']
    var=params['running_var']
    
    out=binarized_linear(input_data,weight, bias, binary_t_f=True)
    out=apply_batch_norm(out, bn_weight, bn_bias,mean=mean,var=var)
    out=apply_activation(out, activation)
    return out


def Create_MLuT_Point(m1,m2,n,switch):#m*mのMLuT行列内のMLuTに0から始まる番号をつける(行列の要素番号で処理)とともにそれらのMLuT行列内の位置を表す行列(縦:m*m,横:2)と縦横共にm*mで，接続されている場合には1,層でない場合には0の行列を作成する
    #switch==0のとき，奇数番目が下に凸(右とのつながりは)，switch==1の時，偶数番目が下に凸(0スタートで左側から数える)
    MLut_point= np.empty((m1*m2, 2), dtype=object)#MLuTの位置情報
    Connecting_Matrix= np.empty((m1*m2, m1*m2), dtype=object)#MLuTの接続情報
    MLuT_Pin_matrix= np.zeros((m1*m2, m1*m2), dtype=object)#MLuTの接続情報
    for i in range(m1):
        for j in range(m2):
            MLut_point[i*m2+j]=(i,j)#(y座標,x座標)
            
            for k in range(m1*m2):
                inout_matrix=np.zeros((4, 4), dtype=object)#MLuTのピン単位の接続情報，縦軸はconnecting_matrixの縦軸のMLuTの入出力(入力側),横軸はconnecting_matrixの縦軸のMLuTの入出力(出力側)で，数字の若い順(0,1,2,3番目の要素)に左上，左下，右下，右上となる
                    
                if (k==i*m2+j-m2) or (k==i*m2+j+m2) or (((k==i*m2+j-m2+(0.5-(switch+i)%2)*2) or (k==i*m2+j+m2+(0.5-(switch+i)%2)*2)) and not((j==0 and (switch+i)%2==1)or(j==m2-1 and (switch+i)%2==0) ) ):
                    Connecting_Matrix[i*m2+j,k]=1
                    
                    #print(i,j,i*m2+j,k,"m",m1,m2)
                    if k==i*m2+j+m2 :
                        if (switch+i)%2==0:
                            inout_matrix[3,1]=1
                        else:
                            inout_matrix[2,0]=1
                    elif k==i*m2+j-m2: 
                        if (switch+i)%2==1:
                            inout_matrix[1,3]=1
                        else:
                            inout_matrix[0,2]=1
                    elif k==i*m2+j-m2-1:
                            inout_matrix[0,2]=1
                    elif k==i*m2+j-m2+1:
                            inout_matrix[1,3]=1
                    elif k==i*m2+j+m2-1:
                            inout_matrix[3,1]=1
                    elif k==i*m2+j+m2+1:
                            inout_matrix[2,0]=1
                else:   
                    Connecting_Matrix[i*m2+j,k]=0
            
                MLuT_Pin_matrix[i*m2+j,k]=inout_matrix
    return MLut_point,Connecting_Matrix,MLuT_Pin_matrix


def mlut_inside(m,n):#m:一方向あたりの入力数,n:入力方向
    G = nx.DiGraph()
    positions = {}
    # 配列データの準備
    parent_data_primal = np.zeros((pow(2,((m*n)//2)),n*m))  # m*n/2入力m*n出力を表す配列データ
    parent_data_inout = np.zeros(n*m)   # n*mbitの配列データ
    
    #親ノードの配置
    parent_nodes=[(0,0),(1,0),(2,0)]#SRAM1，入出力レジスタ，SRAM2
    parent_nodes_name=['SRAM','ioresister','SRAM']
    parent_nodes_id=[1,0,2]
    for index , (x,y) in enumerate(parent_nodes):#SRAMの配置
        parent_node = (x,y)
        positions[parent_node] = (x,y)  # 親ノードの位置
        
        matrix=parent_data_primal if parent_nodes_name[index]=='SRAM' else parent_data_inout
        G.add_node(parent_node, id=parent_nodes_id[index], color='red', size=400,fontsize=10,matrix_data=matrix)
        G.nodes[parent_node]['node_name']=parent_nodes_name[index]
    # 子ノードの作成と親ノードとのエッジの追加
    offsets_low = [(-0.25, i*(2/(2*m))-(2/(4*m))) for i in range(m, -m, -1)]
    offsets_up  = [(0.25, i*(2/(2*m))+(2/(4*m))) for i in range(-m, m, 1)]
    
    for j in range(n//2):
        for i in range(m*2):
            child_id = i + j*m*2
            positions[child_id] = (parent_nodes[0][0]+offsets_low[i][0],parent_nodes[0][1]+offsets_low[i][1]) if j<(n//2)//2 else (parent_nodes[2][0]+offsets_up[i][0],parent_nodes[2][1]+offsets_up[i][1])  # 親ノードの位置
            G.add_node(child_id,id=child_id,size=100,fontsize=7)
            G.nodes[child_id]['node_name']='inout'
            if j<(n//2)//2:
                G.add_edge( child_id,parent_nodes[0])
            else:   
                G.add_edge( child_id,parent_nodes[2])
            G.add_edge(parent_nodes[2], child_id)    
            G.add_edge(parent_nodes[0], child_id)
            G.add_edge(parent_nodes[1], child_id)
                
    nx.set_node_attributes(G, positions, 'pos')

    # ネットワークの描画（必要に応じて）
    #plt.figure(figsize=(8, 8))
    #labels = nx.get_node_attributes(G, 'id')  # ノードIDを取得
    #sizes = [G.nodes[node]['size'] for node in G]  # ノードのサイズを取得
    #nx.draw(G, pos=positions, with_labels=False ,labels=labels, node_color='skyblue', edge_color='gray', node_size=sizes)
        
    #for node, data in G.nodes(data=True):
    #    label = {node: data['id']}
    #    font_size = data['fontsize']
    #    nx.draw_networkx_labels(G, pos=positions, labels=label, font_size=font_size)

    #plt.show()
    return G


def MLuT_network_view_complex(mx,my,n,switch,Connect_matrix):#m:グリッドサイズ#大まかなサイズのMLuT行列の生成
    # ネットワークの初期化
    G = nx.DiGraph()

    
    # ノードの配置と親ノードの追加
    positions = {}
    for i in range(mx):#横
        for j in range(my):#縦
            node_id = my * i + j  # 親ノードにIDを割り当て
            # switchが0の場合は奇数列を0.5ずらす、1の場合は-0.5ずらす
            shift_y = 0.5 if switch == 0 else -0.5
            y_offset = shift_y if i % 2 == 1 else 0
            sub_G=mlut_inside(n,4)#サブネットワークの定義
            parent_node = (i, j)
            positions[parent_node] = (i, -j + y_offset)  # 親ノードの位置
            G.add_node(parent_node, id=node_id, attribute='parent' , color='green', size=400,fontsize=10,sub_network=sub_G)
                
            
            G.nodes[parent_node]['pos_x_MRLD']=i
            G.nodes[parent_node]['pos_y_MRLD']=j
            G.nodes[parent_node]['io_i']=0#0:入出力なし
            G.nodes[parent_node]['io_o']=0#0:入出力なし
            G.nodes[parent_node]['in_way']=[0,0,0,0]#
            G.nodes[parent_node]['out_way']=[0,0,0,0]#
            # 子ノードの追加と位置の割り当て
            offsets = [(-0.25, 0.25), (-0.25, -0.25),  (0.25, -0.25),(0.25, 0.25)]
            child_ids = [0, 1, 2, 3]  # 子ノードに番号を割り当て
            child_sizes = [200, 200, 200, 200]
            for index, (dx, dy) in enumerate(offsets):
                child_node = (i + dx, j + dy)
                positions[child_node] = (i + dx, -j + dy + y_offset)
                G.add_node(child_node,  id=child_ids[index],attribute='child',size=child_sizes[index],fontsize=7)
                G.add_edge(parent_node, child_node)
                G.add_edge(child_node, parent_node)
                #孫ノードの追加と位置の割り当て
                offsets_2=np.zeros((n), dtype=object)
                for of in range(n):
                    offsets_2[of]=((-0.5+index//2)*0.25,(0.5-index//2)*0.3+(0.3/(n-1))*of*(-0.5+index//2)*2)
                #print(offsets_2)
                for index_2, (dx2,dy2) in enumerate(offsets_2):
                    child_node_2 = (i + dx+dx2, j + dy+dy2)
                    # print(dx2,dy2,child_node_2)
                    positions[child_node_2] = (i + dx+dx2, -j + dy+dy2 + y_offset)
                    G.add_node(child_node_2,  id=index*n+index_2,attribute='grandchild',size=50,fontsize=4)
                    G.add_edge(child_node_2, child_node)
                    G.add_edge(child_node, child_node_2)

    #print("m",mx,my)
    for i in range(mx*my):#横
        for j in range(mx*my):#縦
            if (np.any(Connect_matrix[i,j]!=0)):
                connect=Connect_matrix[i,j]
                parent_node_1,parent_node_2=(i//my,i%my),(j//my,j%my)
                
                #print((i//my,i%my),(j//my,j%my),i,j)
                first_node_children,second_node_children = list(G.successors(parent_node_1)),list(G.successors(parent_node_2))
                #print(first_node_children,second_node_children)
                for k in range(4):#横
                    for l in range(4):#縦
                        if connect[l,k]==1:
                            first_node_child,second_node_child=first_node_children[l],second_node_children[k]
                            first_g_children,second_g_children = list(G.successors(first_node_child)),list(G.successors(second_node_child))
                            # print(first_g_children,second_g_children,first_node_child,second_node_child,"###")
                            for o in range(n):
                                G.add_edge(first_g_children[o+1],second_g_children[n-o])

                            
            
    # 描画
    plt.figure(figsize=(8, 8))
    labels = nx.get_node_attributes(G, 'id')  # ノードIDを取得
    sizes = [G.nodes[node]['size'] for node in G]  # ノードのサイズを取得
    nx.draw(G, pos=positions, with_labels=False ,labels=labels, node_color='skyblue', edge_color='gray', node_size=sizes)
        
    for node, data in G.nodes(data=True):
        label = {node: data['id']}
        font_size = data['fontsize']
        nx.draw_networkx_labels(G, pos=positions, labels=label, font_size=font_size)

    plt.show()
    #id0番のサブネットワークの取り出し
    node_1 = [node for node, attr in G.nodes(data=True) if (attr.get('id') == 0) and (attr.get('attribute') == 'parent')]

    subnetwork_1=G.nodes[node_1[0]]['sub_network']
    subnetwork_nodes_position=nx.get_node_attributes(subnetwork_1, 'pos')
    G.graph['layer_num_x']=mx
    G.graph['layer_num_y']=my
    G.graph['MRLD_input_num']=4*n
    G.graph['Flip_up']=switch
    
    
    # print(node_1)
    # print(subnetwork_1)
    # print()
    # print()
    #サブネットワークの描画
    plt.figure(figsize=(8, 8))
    labels = nx.get_node_attributes(subnetwork_1, 'id')  # ノードIDを取得
    sizes = [subnetwork_1.nodes[node]['size'] for node in subnetwork_1]  # ノードのサイズを取得
    nx.draw(subnetwork_1, pos=subnetwork_nodes_position, with_labels=False ,labels=labels, node_color='skyblue', edge_color='gray', node_size=sizes)
        
    for node, data in subnetwork_1.nodes(data=True):
        label = {node: data['id']}
        font_size = data['fontsize']
        nx.draw_networkx_labels(subnetwork_1, pos=subnetwork_nodes_position, labels=label, font_size=font_size)
    
    plt.show()
    nx.set_node_attributes(G, positions, 'pos')
    return G


def find_nodes(nodes, attribute_name, attribute_value):
    for node, attr in nodes(data=True):
        if attr.get(attribute_name) == attribute_value:
            yield node


def MNN_insert_to_MRP_comp(network_MLUT,network_MNN,y_shift):
    network_MLUT_MNN=network_MLUT
    #MLUT_net
    layer_num=network_MNN.graph["layer_num"]
    input_sizes=network_MNN.graph["input_output_types"]['input_counts']
    output_sizes=network_MNN.graph["input_output_types"]['output_counts']
    n=network_MLUT.graph['MRLD_input_num']
    output_shape_MRP=[]
    for node in find_nodes(network_MLUT_MNN.nodes,'attribute','parent'):#全MLuTの配列の初期化
        node_name=next(find_nodes(network_MLUT_MNN.nodes[node]['sub_network'].nodes,'id',1),None)
        
        network_MLUT_MNN.nodes[node]['sub_network'].nodes[node_name]['matrix_data']=torch.zeros((2**(n//2),n),dtype=torch.int32)
        
        node_name=next(find_nodes(network_MLUT_MNN.nodes[node]['sub_network'].nodes,'id',2),None)
        
        network_MLUT_MNN.nodes[node]['sub_network'].nodes[node_name]['matrix_data']=torch.zeros((2**(n//2),n),dtype=torch.int32)
        
        node_name=next(find_nodes(network_MLUT_MNN.nodes[node]['sub_network'].nodes,'id',0),None)
        
        network_MLUT_MNN.nodes[node]['sub_network'].nodes[node_name]['matrix_data']=torch.zeros((n),dtype=torch.int32)#全部を0にする(出力は全部適用される)
        
    
    for node in find_nodes(network_MNN.nodes,'layer','main'): 
        #print(node,network_MNN.nodes[node]['pos_x'],"    ",network_MNN.nodes[node]['pos_x_MRLD'],"   ",network_MNN.nodes[node]['pos_y'],"$%%$")
        #sub_net = network_MNN.nodes[node].get('sub_network', None)
        target_x_=network_MNN.nodes[node]['pos_x']
        target_x_MRLD=network_MNN.nodes[node]['pos_x_MRLD']
        target_y_MRLD=network_MNN.nodes[node]['pos_y_MRLD']
        target_node_pos=(target_x_MRLD,target_y_MRLD+y_shift)
        output_port = network_MNN.nodes[node]['output_port']
        input_range = network_MNN.nodes[node]['io_in']
        output_range = network_MNN.nodes[node]['io_out']
        start_index = output_port[0]
        end_index = output_port[-1]

            
        print(target_x_MRLD,target_x_MRLD,start_index,end_index,output_port,input_range,output_range)
        if start_index>end_index:
            buff=start_index
            start_index=end_index
            end_index=buff
        if min(network_MNN.nodes[node]['input_port']) < max(input_sizes):
            #print(network_MLUT_MNN.nodes[target_node_pos]['sub_network'].nodes[(0, 0)])
            node_name=next(find_nodes(network_MLUT_MNN.nodes[target_node_pos]['sub_network'].nodes,'id',1),None)
            network_MLUT_MNN.nodes[target_node_pos]['sub_network'].nodes[node_name]['matrix_data']=network_MNN.nodes[node]['Truth_Table_MRLD']
            # print("$$%$$",node,target_node_pos,target_x_MRLD,target_y_MRLD,target_x_)        
            # # 列の数を取得
            # num_columns = len(network_MNN.nodes[node]['Truth_Table_MRLD'][0])
            # print(num_columns)
            # # すべて0の列を探す
            # zero_columns = []
            # for col in range(num_columns):
            #     if all(row[col] == 0 for row in network_MNN.nodes[node]['Truth_Table_MRLD']):
            #         zero_columns.append(col)
            
            # print(f"すべて0の列: {zero_columns}")
        else:
            node_name=next(find_nodes(network_MLUT_MNN.nodes[target_node_pos]['sub_network'].nodes,'id',2),None)
            network_MLUT_MNN.nodes[target_node_pos]['sub_network'].nodes[node_name]['matrix_data']=network_MNN.nodes[node]['Truth_Table_MRLD']
            # print("$$#$$#$$",node,target_node_pos,target_x_MRLD,target_y_MRLD,target_x_)
            

        if target_x_MRLD == 0 or target_x_MRLD == layer_num-1:#MNNの中の外部入力部分と外部出力部分を設定する(実際には外部入力部分しか使用しない) y方向の恥に関しては特に対応していないのでそこを使用する場合には新しく作りたい
            if target_x_==0:
                network_MLUT_MNN.nodes[target_node_pos]['io_i']=1#1:入力側
                if target_x_MRLD == 0:
                    if min(input_sizes)==input_range:
                        if target_y_MRLD==0:#上側
                            network_MLUT_MNN.nodes[target_node_pos]['in_way']=[0,1,0,0]
                        else:
                            network_MLUT_MNN.nodes[target_node_pos]['in_way']=[1,0,0,0]
                        
                    else:
                        network_MLUT_MNN.nodes[target_node_pos]['in_way']=[1,1,0,0]#1:左上,2:左下,4:右下,8:右上
                else:
                    if min(input_sizes)==input_range:
                        if target_y_MRLD==0:#上側
                            network_MLUT_MNN.nodes[target_node_pos]['in_way']=[0,0,1,0]
                        else:
                            network_MLUT_MNN.nodes[target_node_pos]['in_way']=[0,0,0,1]
                            
                    else:
                        network_MLUT_MNN.nodes[target_node_pos]['in_way']=[0,0,1,1]#
            else:
                network_MLUT_MNN.nodes[target_node_pos]['io_o']=1#1:出力側
                if target_x_MRLD == 0:#左側の場合
                    if min(output_sizes)==output_range:
                        if output_port[0]//(n//4)==0:
                            network_MLUT_MNN.nodes[target_node_pos]['out_way']=[1,0,0,0]#1:左上,2:左下,4:右下,8:右上
                        else:
                            network_MLUT_MNN.nodes[target_node_pos]['out_way']=[0,1,0,0]#1:左上,2:左下,4:右下,8:右上
                    else:
                        network_MLUT_MNN.nodes[target_node_pos]['out_way']=[1,1,0,0]
                else:#右側の場合
                    if min(output_sizes)==output_range:
                        if output_port[0]//(n//4)==2:
                            network_MLUT_MNN.nodes[target_node_pos]['out_way']=[0,0,1,0]#1:左上,2:左下,4:右下,8:右上
                        else:
                            network_MLUT_MNN.nodes[target_node_pos]['out_way']=[0,0,0,1]#1:左上,2:左下,4:右下,8:右上
                    else:
                        network_MLUT_MNN.nodes[target_node_pos]['out_way']=[0,0,1,1]
                output_shape_MRP.append(len(output_port))
        else :      
            network_MLUT_MNN.nodes[target_node_pos]['io_o']=0#0:入出力なし
            network_MLUT_MNN.nodes[target_node_pos]['io_i']=0#0:入出力なし
            network_MLUT_MNN.nodes[target_node_pos]['in_way']=[0,0,0,0]#
            network_MLUT_MNN.nodes[target_node_pos]['out_way']=[0,0,0,0]#
        
        
        node_name=next(find_nodes(network_MLUT_MNN.nodes[target_node_pos]['sub_network'].nodes,'id',0),None)
        network_MLUT_MNN.nodes[target_node_pos]['sub_network'].nodes[node_name]['matrix_data'][start_index:end_index+1]=0

    
    network_MLUT_MNN.graph['input_Layer']=network_MNN.graph['input_Layer']
    network_MLUT_MNN.graph['output_Layer']=network_MNN.graph['output_Layer']
    network_MLUT_MNN.graph['output_Layer_shape']=output_shape_MRP
    
    
    # for node in network_MLUT_MNN.nodes: 
    #     # print(node,"$%&%$")
    #     sub_net = network_MLUT_MNN.nodes[node].get('sub_network', None)
    #     if sub_net:
    #         print("\n\n",node)
    #         print(sub_net.nodes[next(find_nodes(sub_net.nodes,'id',0),None)],"$&%&$")
    network_MLUT_MNN.graph['layers_num']=layer_num
    return network_MLUT_MNN


@dataclass
class NeutralMRPMLUT:
    x: int
    y: int
    enabled: bool
    bank0_table: List[List[int]]
    bank1_table: List[List[int]]
    reg_data: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NeutralMRPNetwork:
    format: str
    version: str
    mx: int
    my: int
    n: int
    flip_up: int
    mluts: List[NeutralMRPMLUT]
    flow_direction: str = "left_to_right"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path, indent: int = 2) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=indent)

    @classmethod
    def load_json(cls, path: str | Path) -> "NeutralMRPNetwork":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        mluts = [NeutralMRPMLUT(**m) for m in data["mluts"]]

        return cls(
            format=data["format"],
            version=data["version"],
            mx=data["mx"],
            my=data["my"],
            n=data["n"],
            flip_up=data["flip_up"],
            mluts=mluts,
            flow_direction=data.get(
                "flow_direction",
                data.get("metadata", {}).get("flow_direction", "left_to_right"),
            ),
            metadata=data.get("metadata", {}),
        )


def find_node_by_id(nodes, target_id: int):
    for node, attr in nodes(data=True):
        if attr.get("id") == target_id:
            return node
    raise KeyError(f"sub_network node with id={target_id} not found")


def tensor_to_int_list(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().int().tolist()
    if isinstance(data, np.ndarray):
        return data.astype(int).tolist()
    if isinstance(data, list):
        return data
    raise TypeError(f"Unsupported matrix_data type: {type(data)}")


def is_all_zero(data) -> bool:
    arr = torch.tensor(data)
    return bool(torch.all(arr == 0))


def export_mrp_network_to_neutral(
    model_MNN_in_MRP,
    *,
    out_path: str | Path,
    mx: int,
    my: int,
    n: int,
    flip_up: int,
    flow_direction: str = "left_to_right",
    version: str = "1.0",
) -> NeutralMRPNetwork:
    """
    MNN_insert_to_MRP_comp 後の MLUT_model_in_MNN_advance を
    Vivado 非依存の JSON に保存する。
    """

    mluts: List[NeutralMRPMLUT] = []

    for x in range(mx):
        for y in range(my):
            node = (x, y)

            if node not in model_MNN_in_MRP.nodes:
                raise KeyError(f"MLUT node {node} not found in model_MNN_in_MRP")

            sub_net = model_MNN_in_MRP.nodes[node].get("sub_network", None)
            if sub_net is None:
                raise KeyError(f"MLUT node {node} has no sub_network")

            reg_node = find_node_by_id(sub_net.nodes, 0)
            bank0_node = find_node_by_id(sub_net.nodes, 1)
            bank1_node = find_node_by_id(sub_net.nodes, 2)

            reg_data = tensor_to_int_list(
                sub_net.nodes[reg_node]["matrix_data"]
            )
            bank0_table = tensor_to_int_list(
                sub_net.nodes[bank0_node]["matrix_data"]
            )
            bank1_table = tensor_to_int_list(
                sub_net.nodes[bank1_node]["matrix_data"]
            )

            enabled = not (
                is_all_zero(reg_data)
                and is_all_zero(bank0_table)
                and is_all_zero(bank1_table)
            )

            mluts.append(
                NeutralMRPMLUT(
                    x=x,
                    y=y,
                    enabled=enabled,
                    bank0_table=bank0_table,
                    bank1_table=bank1_table,
                    reg_data=reg_data,
                    metadata={
                        "source_node": str(node),
                    },
                )
            )

    neutral = NeutralMRPNetwork(
        format="mlut_mrp_neutral",
        version=version,
        mx=mx,
        my=my,
        n=n,
        flip_up=flip_up,
        mluts=mluts,
        flow_direction=flow_direction,
        metadata={
            "flow_direction": flow_direction,
            "bank0": "sub_network node id=1, left input SRAM",
            "bank1": "sub_network node id=2, right input SRAM",
            "reg": "sub_network node id=0",
        },
    )

    neutral.save_json(out_path)
    save_io_layers(model_MNN_in_MRP)
    return neutral


def validate_neutral_mrp_network(net: NeutralMRPNetwork) -> List[str]:
    errors: List[str] = []

    if net.mx <= 0:
        errors.append("mx must be positive")
    if net.my <= 0:
        errors.append("my must be positive")
    if net.n <= 0:
        errors.append("n must be positive")

    expected_count = net.mx * net.my
    if len(net.mluts) != expected_count:
        errors.append(
            f"MLUT count mismatch: expected {expected_count}, got {len(net.mluts)}"
        )

    seen = set()

    for m in net.mluts:
        pos = (m.x, m.y)

        if pos in seen:
            errors.append(f"duplicated MLUT position: {pos}")
        seen.add(pos)

        if not (0 <= m.x < net.mx):
            errors.append(f"x out of range: {pos}")
        if not (0 <= m.y < net.my):
            errors.append(f"y out of range: {pos}")

        if len(m.reg_data) != 4 * net.n:
            errors.append(
                f"reg_data width mismatch at {pos}: "
                f"expected {4 * net.n}, got {len(m.reg_data)}"
            )

        expected_rows = 2 ** (2 * net.n)

        if len(m.bank0_table) != expected_rows:
            errors.append(
                f"bank0 row mismatch at {pos}: "
                f"expected {expected_rows}, got {len(m.bank0_table)}"
            )

        if len(m.bank1_table) != expected_rows:
            errors.append(
                f"bank1 row mismatch at {pos}: "
                f"expected {expected_rows}, got {len(m.bank1_table)}"
            )

        for row_i, row in enumerate(m.bank0_table):
            if len(row) != 4 * net.n:
                errors.append(
                    f"bank0 width mismatch at {pos}, row {row_i}: "
                    f"expected {4 * net.n}, got {len(row)}"
                )
                break

        for row_i, row in enumerate(m.bank1_table):
            if len(row) != 4 * net.n:
                errors.append(
                    f"bank1 width mismatch at {pos}, row {row_i}: "
                    f"expected {4 * net.n}, got {len(row)}"
                )
                break

    return errors


def save_io_layers(model_MNN_in_MRP, out_dir="export_io_layers"):
    from pathlib import Path
    import torch
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_layer = model_MNN_in_MRP.graph["input_Layer"]
    output_layer = model_MNN_in_MRP.graph["output_Layer"]

    torch.save(input_layer, out_dir / "input_layer.pt")
    torch.save(output_layer, out_dir / "output_layer.pt")

    meta = {
        "input_layer_file": "input_layer.pt",
        "output_layer_file": "output_layer.pt",
        "note": "PyTorch module保存。Vivado直接読込用ではなく、後段でparams/table変換するための中間保存。"
    }

    with open(out_dir / "io_layers_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
