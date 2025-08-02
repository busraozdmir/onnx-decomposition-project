import onnx
import numpy as np
import onnx_graphsurgeon as OGS

from onnx import shape_inference
import copy

import os
import sys

from google.protobuf.json_format import MessageToDict

from onnx import helper, numpy_helper, TensorProto

# from kernel_profiler import get_input_node_info
def get_input_node_info(onnx_model):
    # TVM from_onnx() requires shape_dict to be a dictionary of node name: List of dimensions
    shape_dict = {}
    input_name = ""
    DTYPE = ""
    input_shape = []
    
    for _input in onnx_model.graph.input:
        # ONNX format returns graph nodes as protobuf object
        m_dict = MessageToDict(_input)
        print("input_name : ", m_dict['name'])
        print("input_shape: ", m_dict["type"]["tensorType"]['shape'])
        print("input_dtype: ", m_dict["type"]["tensorType"]['elemType'])
        dim_info = m_dict["type"]["tensorType"]["shape"]["dim"]
        input_shape = [int(d.get("dimValue")) for d in dim_info]
        input_name = m_dict["name"]
        shape_dict[input_name] = input_shape
        
        # TODO: Convert enum elemType to required datatype
        DTYPE = "float32" if m_dict["type"]["tensorType"]['elemType'] == 1 else "float32"
        
    return shape_dict, input_name, input_shape, DTYPE


# Reference: https://github.com/NVIDIA/TensorRT/blob/master/tools/onnx-graphsurgeon/examples/08_replacing_a_subgraph/replace.py

#%%
def create_test_softmax_subgraph(softmax_axis=-1):
    a = OGS.Variable("a", shape=[3,3], dtype=np.float32)
    
    output = OGS.Variable("output", shape=[3,3], dtype=np.float32)
    
    c1 = OGS.Node(
                op = "Softmax",
                inputs=[a],
                outputs=[output],
                attrs={'axis': softmax_axis},
                name="Softmax"
            )
        
    OGS_graph = OGS.Graph(nodes=[c1], inputs=[a], outputs=[output])
    onnx_graph = OGS.export_onnx(OGS_graph)
    onnx_graph_shape_inferred = shape_inference.infer_shapes(onnx_graph)
    
    return onnx_graph_shape_inferred



def create_test_instancenorm_subgraph(epsilon=1e-5):

    x = OGS.Variable("x", shape=[1, 2, 1, 3], dtype=np.float32)
    s = OGS.Variable("s", shape=[2,], dtype=np.float32)
    bias = OGS.Variable("bias", shape=[2,], dtype=np.float32)
    
    # eps_val = OGS.Constant("eps_val", values=np.array([epsilon]))


    output = OGS.Variable("output", shape=[1, 2, 1, 3], dtype=np.float32)
    
    c1 = OGS.Node(
                op = "InstanceNormalization",
                inputs=[x, s, bias],
                outputs=[output],
                attrs={'epsilon': epsilon},
                name="InstanceNormalization"
            )
        
    OGS_graph = OGS.Graph(nodes=[c1], inputs=[x, s, bias], outputs=[output])
    onnx_graph = OGS.export_onnx(OGS_graph)
    onnx_graph_shape_inferred = shape_inference.infer_shapes(onnx_graph)
    
    return onnx_graph_shape_inferred



def create_test_subgraph(a_tensor, b_tensor, c_tensor, out_tensor_shape=[1, 1, 65536, 32], softmax_axis=-1):
    a = OGS.Variable("a", shape=list(a_tensor.size()), dtype=np.float32)
    b = OGS.Variable("b", shape=list(b_tensor.size()), dtype=np.float32)
    c = OGS.Variable("c", shape=list(c_tensor.size()), dtype=np.float32)
    
    div_out = OGS.Variable("div_out", dtype=np.float32)
    softmax_out = OGS.Variable("softmax_out", dtype=np.float32)
    
    output = OGS.Variable("output", shape=out_tensor_shape, dtype=np.float32)
    
    c0 = OGS.Node(
                op = "Div",
                inputs=[a, b],
                outputs=[div_out],
                name="Div"
            )
    
    c1 = OGS.Node(
                op = "Softmax",
                inputs=[div_out],
                outputs=[softmax_out],
                attrs={'axis': softmax_axis},
                name="Softmax"
            )
    
    c2 = OGS.Node(
                op = "MatMul",
                inputs=[softmax_out, c],
                outputs=[output],
                name="MatMul"
            )
        
    OGS_graph = OGS.Graph(nodes=[c0, c1, c2], inputs=[a, b, c], outputs=[output])
    onnx_graph = OGS.export_onnx(OGS_graph)
    onnx_graph_shape_inferred = shape_inference.infer_shapes(onnx_graph)
    
    return onnx_graph_shape_inferred


def make_scalar(name, value, dtype=TensorProto.FLOAT):
    return helper.make_tensor(name=name, data_type=dtype, dims=[], vals=[value])

def fission_relu(graph, node, subblock_name, counter, counter_dict):
    counter_dict["Relu"] = counter_dict.get("Relu", 0) + 1
    x = node.inputs[0]
    y = node.outputs[0]

    name_prefix = f"{subblock_name}_relu_fission_{counter}_"

    zero_const = OGS.Constant(name_prefix + "zero", values=np.array([0.0], dtype=np.float32))
    maxn = OGS.Variable(name_prefix + "max", dtype=np.float32)

    graph.nodes.append(
        OGS.Node(op="Max", inputs=[x, zero_const], outputs=[maxn], name=name_prefix + "Max")
    )

    for other_node in graph.nodes:
        for i, inp in enumerate(other_node.inputs):
            if inp.name == y.name:
                other_node.inputs[i] = maxn

    for i, out in enumerate(graph.outputs):
        if out.name == y.name:
            graph.outputs[i] = maxn

    node.inputs.clear()
    node.outputs.clear()

def fission_dropout(graph, node, subblock_name, counter, counter_dict):
    counter_dict["Dropout"] = counter_dict.get("Dropout", 0) + 1
    x = node.inputs[0]
    y = node.outputs[0]

    name_prefix = f"{subblock_name}_dropout_fission_{counter}_"
    identity = OGS.Variable(name_prefix + "identity", dtype=np.float32)

    graph.nodes.append(
        OGS.Node(op="Identity", inputs=[x], outputs=[identity], name=name_prefix + "Identity")
    )

    for other_node in graph.nodes:
        for i, inp in enumerate(other_node.inputs):
            if inp.name == y.name:
                other_node.inputs[i] = identity

    for i, out in enumerate(graph.outputs):
        if out.name == y.name:
            graph.outputs[i] = identity

    node.inputs.clear()
    node.outputs.clear()

def fission_reciprocal(graph, node, subblock_name, counter, counter_dict):
    counter_dict["Reciprocal"] = counter_dict.get("Reciprocal", 0) + 1
    x = node.inputs[0]
    y = node.outputs[0]

    name_prefix = f"{subblock_name}_reciprocal_fission_{counter}_"

    one_const = OGS.Constant(name_prefix + "one", values=np.array([1.0], dtype=np.float32))
    div = OGS.Variable(name_prefix + "div", dtype=np.float32)

    graph.nodes.append(
        OGS.Node(op="Div", inputs=[one_const, x], outputs=[div], name=name_prefix + "Div")
    )

    for other_node in graph.nodes:
        for i, inp in enumerate(other_node.inputs):
            if inp.name == y.name:
                other_node.inputs[i] = div

    for i, out in enumerate(graph.outputs):
        if out.name == y.name:
            graph.outputs[i] = div

    node.inputs.clear()
    node.outputs.clear()

def fission_mish(graph, node, subblock_name, counter, counter_dict):
    counter_dict["Mish"] = counter_dict.get("Mish", 0) + 1
    x = node.inputs[0]
    y = node.outputs[0]

    name_prefix = f"{subblock_name}_mish_fission_{counter}_"

    sp = OGS.Variable(name_prefix + "softplus", dtype=np.float32)
    t = OGS.Variable(name_prefix + "tanh", dtype=np.float32)
    mul = OGS.Variable(name_prefix + "mul", dtype=np.float32)

    graph.nodes.extend([
        OGS.Node(op="Softplus", inputs=[x], outputs=[sp], name=name_prefix + "Softplus"),
        OGS.Node(op="Tanh", inputs=[sp], outputs=[t], name=name_prefix + "Tanh"),
        OGS.Node(op="Mul", inputs=[x, t], outputs=[mul], name=name_prefix + "Mul"),
    ])

    for other_node in graph.nodes:
        for i, inp in enumerate(other_node.inputs):
            if inp.name == y.name:
                other_node.inputs[i] = mul

    for i, out in enumerate(graph.outputs):
        if out.name == y.name:
            graph.outputs[i] = mul

    node.inputs.clear()
    node.outputs.clear()

def fission_layernorm(graph, x, scale, bias, epsilon, subblock_name, orig_node, counter_dict):
    counter_dict["LayerNormalization"] = counter_dict.get("LayerNormalization", 0) + 1
    mean_axes = [len(x.shape) - 1]

    name_prefix = subblock_name + "_layernorm_fission_"

    broadcast_shape_vals = np.array([1] * (len(x.shape) - 1) + [x.shape[-1]], dtype=np.int64)
    shape_const = OGS.Constant(name_prefix + "broadcast_shape", values=broadcast_shape_vals)

    scale_reshaped = OGS.Variable(name_prefix + "scale_reshaped", dtype=np.float32, shape=broadcast_shape_vals.tolist())
    bias_reshaped = OGS.Variable(name_prefix + "bias_reshaped", dtype=np.float32, shape=broadcast_shape_vals.tolist())

    mean_out = OGS.Variable(name_prefix + "mean_out", dtype=np.float32)
    sub = OGS.Variable(name_prefix + "centered", dtype=np.float32)
    square = OGS.Variable(name_prefix + "squared", dtype=np.float32)
    var = OGS.Variable(name_prefix + "var", dtype=np.float32)
    eps_val = OGS.Constant(name_prefix + "eps", values=np.array([epsilon], dtype=np.float32))
    var_eps = OGS.Variable(name_prefix + "var_eps", dtype=np.float32)
    std = OGS.Variable(name_prefix + "std", dtype=np.float32)
    norm = OGS.Variable(name_prefix + "norm", dtype=np.float32)
    scaled = OGS.Variable(name_prefix + "scaled", dtype=np.float32)
    shifted = OGS.Variable(name_prefix + "shifted", dtype=np.float32)

    graph.nodes.extend([
        OGS.Node("Reshape", inputs=[scale, shape_const], outputs=[scale_reshaped], name=name_prefix + "ReshapeScale"),
        OGS.Node("Reshape", inputs=[bias, shape_const], outputs=[bias_reshaped], name=name_prefix + "ReshapeBias"),
        OGS.Node("ReduceMean", inputs=[x], outputs=[mean_out], attrs={"axes": mean_axes, "keepdims": 1}, name=name_prefix + "ReduceMean"),
        OGS.Node("Sub", inputs=[x, mean_out], outputs=[sub], name=name_prefix + "Sub"),
        OGS.Node("Mul", inputs=[sub, sub], outputs=[square], name=name_prefix + "Square"),
        OGS.Node("ReduceMean", inputs=[square], outputs=[var], attrs={"axes": mean_axes, "keepdims": 1}, name=name_prefix + "ReduceMeanVar"),
        OGS.Node("Add", inputs=[var, eps_val], outputs=[var_eps], name=name_prefix + "AddEps"),
        OGS.Node("Sqrt", inputs=[var_eps], outputs=[std], name=name_prefix + "Sqrt"),
        OGS.Node("Div", inputs=[sub, std], outputs=[norm], name=name_prefix + "Normalize"),
        OGS.Node("Mul", inputs=[norm, scale_reshaped], outputs=[scaled], name=name_prefix + "Scale"),
        OGS.Node("Add", inputs=[scaled, bias_reshaped], outputs=[shifted], name=name_prefix + "Shift"),
    ])

    shifted.name = orig_node.outputs[0].name
    shifted.shape = orig_node.outputs[0].shape

    return shifted

def fission_hardsigmoid(graph, node, subblock_name, counter, counter_dict):
    counter_dict["HardSigmoid"] = counter_dict.get("HardSigmoid", 0) + 1
    x = node.inputs[0]
    y = node.outputs[0]

    alpha = node.attrs.get("alpha", 0.2)
    beta = node.attrs.get("beta", 0.5)

    name_prefix = f"{subblock_name}_hardsigmoid_fission_{counter}_"

    alpha_const = OGS.Constant(name_prefix + "alpha", values=np.array([alpha], dtype=np.float32))
    beta_const = OGS.Constant(name_prefix + "beta", values=np.array([beta], dtype=np.float32))
    zero_const = OGS.Constant(name_prefix + "zero", values=np.array([0.0], dtype=np.float32))
    one_const = OGS.Constant(name_prefix + "one", values=np.array([1.0], dtype=np.float32))

    mul = OGS.Variable(name_prefix + "mul", dtype=np.float32)
    add = OGS.Variable(name_prefix + "add", dtype=np.float32)
    clip = OGS.Variable(name_prefix + "clip", dtype=np.float32)

    graph.nodes.extend([
        OGS.Node(op="Mul", inputs=[x, alpha_const], outputs=[mul], name=name_prefix + "Mul"),
        OGS.Node(op="Add", inputs=[mul, beta_const], outputs=[add], name=name_prefix + "Add"),
        OGS.Node(op="Clip", inputs=[add, zero_const, one_const], outputs=[clip], name=name_prefix + "Clip")
    ])

    # Yeni oluşturulan `clip` değişkenini eski output'un yerine koy
    for other_node in graph.nodes:
        for i, input_tensor in enumerate(other_node.inputs):
            if input_tensor.name == y.name:
                other_node.inputs[i] = clip

    # Eğer bu node doğrudan output ise, değiştir
    for i, output in enumerate(graph.outputs):
        if output.name == y.name:
            graph.outputs[i] = clip

    node.inputs.clear()
    node.outputs.clear()

def fission_hardswish(graph, node, subblock_name, counter,counter_dict):
    counter_dict["HardSwish"] = counter_dict.get("HardSwish", 0) + 1
    x = node.inputs[0]
    y = node.outputs[0]

    name_prefix = f"{subblock_name}_hardswish_fission_{counter}_"

    const_three = OGS.Constant(name_prefix + "three", values=np.array([3.0], dtype=np.float32))
    const_six = OGS.Constant(name_prefix + "six", values=np.array([6.0], dtype=np.float32))
    const_zero = OGS.Constant(name_prefix + "zero", values=np.array([0.0], dtype=np.float32))

    add = OGS.Variable(name_prefix + "add", dtype=np.float32)
    clip = OGS.Variable(name_prefix + "clip", dtype=np.float32)
    mul = OGS.Variable(name_prefix + "mul", dtype=np.float32)
    div = OGS.Variable(name_prefix + "div", dtype=np.float32)

    graph.nodes.extend([
        OGS.Node(op="Add", inputs=[x, const_three], outputs=[add], name=name_prefix + "Add"),
        OGS.Node(op="Clip", inputs=[add, const_zero, const_six], outputs=[clip], name=name_prefix + "Clip"),
        OGS.Node(op="Mul", inputs=[x, clip], outputs=[mul], name=name_prefix + "Mul"),
        OGS.Node(op="Div", inputs=[mul, const_six], outputs=[div], name=name_prefix + "Div")
    ])


    for other_node in graph.nodes:
        for i, inp in enumerate(other_node.inputs):
            if inp.name == y.name:
                other_node.inputs[i] = div

    for i, out in enumerate(graph.outputs):
        if out.name == y.name:
            graph.outputs[i] = div

    node.inputs.clear()
    node.outputs.clear()

def fission_sigmoid(graph, node, subblock_name, counter, counter_dict):
    counter_dict["Sigmoid"] = counter_dict.get("Sigmoid", 0) + 1
    x = node.inputs[0]
    y = node.outputs[0]

    name_prefix = f"{subblock_name}_sigmoid_fission_{counter}_"

    one_const = OGS.Constant(name_prefix + "one", values=np.array([1.0], dtype=np.float32))

    neg = OGS.Variable(name_prefix + "neg", dtype=np.float32)
    exp = OGS.Variable(name_prefix + "exp", dtype=np.float32)
    denom = OGS.Variable(name_prefix + "denom", dtype=np.float32)
    div = OGS.Variable(name_prefix + "div", dtype=np.float32)

    graph.nodes.extend([
        OGS.Node(op="Neg", inputs=[x], outputs=[neg], name=name_prefix + "Neg"),
        OGS.Node(op="Exp", inputs=[neg], outputs=[exp], name=name_prefix + "Exp"),
        OGS.Node(op="Add", inputs=[exp, one_const], outputs=[denom], name=name_prefix + "Add"),
        OGS.Node(op="Div", inputs=[one_const, denom], outputs=[div], name=name_prefix + "Div")
    ])

    # Tüm graph'ta sigmoid'in çıktısını kullanan yerleri güncelle
    for other_node in graph.nodes:
        for i, inp in enumerate(other_node.inputs):
            if inp.name == y.name:
                other_node.inputs[i] = div

    for i, out in enumerate(graph.outputs):
        if out.name == y.name:
            graph.outputs[i] = div

    node.inputs.clear()
    node.outputs.clear()

def fission_tanh(graph, node, subblock_name, counter, counter_dict):
    counter_dict["Tanh"] = counter_dict.get("Tanh", 0) + 1
    x = node.inputs[0]
    y = node.outputs[0]

    name_prefix = f"{subblock_name}_tanh_fission_{counter}_"

    neg = OGS.Variable(name_prefix + "neg", dtype=np.float32)
    exp_p = OGS.Variable(name_prefix + "exp_p", dtype=np.float32)
    exp_n = OGS.Variable(name_prefix + "exp_n", dtype=np.float32)
    num = OGS.Variable(name_prefix + "num", dtype=np.float32)
    denom = OGS.Variable(name_prefix + "denom", dtype=np.float32)
    div = OGS.Variable(name_prefix + "div", dtype=np.float32)

    graph.nodes.extend([
        OGS.Node(op="Neg", inputs=[x], outputs=[neg], name=name_prefix + "Neg"),
        OGS.Node(op="Exp", inputs=[x], outputs=[exp_p], name=name_prefix + "ExpP"),
        OGS.Node(op="Exp", inputs=[neg], outputs=[exp_n], name=name_prefix + "ExpN"),
        OGS.Node(op="Sub", inputs=[exp_p, exp_n], outputs=[num], name=name_prefix + "Sub"),
        OGS.Node(op="Add", inputs=[exp_p, exp_n], outputs=[denom], name=name_prefix + "Add"),
        OGS.Node(op="Div", inputs=[num, denom], outputs=[div], name=name_prefix + "Div"),
    ])

    for other_node in graph.nodes:
        for i, inp in enumerate(other_node.inputs):
            if inp.name == y.name:
                other_node.inputs[i] = div

    for i, out in enumerate(graph.outputs):
        if out.name == y.name:
            graph.outputs[i] = div

    node.inputs.clear()
    node.outputs.clear()

def fission_leaky_relu(graph, node, subblock_name, counter,counter_dict):
    counter_dict["LeakyRelu"] = counter_dict.get("LeakyRelu", 0) + 1
    x = node.inputs[0]
    y = node.outputs[0]

    alpha = node.attrs.get("alpha", 0.01)
    name_prefix = f"{subblock_name}_leakyrelu_fission_{counter}_"

    alpha_const = OGS.Constant(name_prefix + "slope", values=np.array([alpha], dtype=np.float32))
    mul = OGS.Variable(name_prefix + "mul", dtype=np.float32)
    maxn = OGS.Variable(name_prefix + "max", dtype=np.float32)

    graph.nodes.extend([
        OGS.Node(op="Mul", inputs=[x, alpha_const], outputs=[mul], name=name_prefix + "Mul"),
        OGS.Node(op="Max", inputs=[x, mul], outputs=[maxn], name=name_prefix + "Max")
    ])

    for other_node in graph.nodes:
        for i, inp in enumerate(other_node.inputs):
            if inp.name == y.name:
                other_node.inputs[i] = maxn

    for i, out in enumerate(graph.outputs):
        if out.name == y.name:
            graph.outputs[i] = maxn

    node.inputs.clear()
    node.outputs.clear()

def fission_clip(graph, node, subblock_name, counter, counter_dict):
    counter_dict["Clip"] = counter_dict.get("Clip", 0) + 1
    x = node.inputs[0]
    y = node.outputs[0]

    min_val = node.attrs.get("min", 0.0)
    max_val = node.attrs.get("max", 1.0)

    name_prefix = f"{subblock_name}_clip_fission_{counter}_"

    min_const = OGS.Constant(name_prefix + "min", values=np.array([min_val], dtype=np.float32))
    max_const = OGS.Constant(name_prefix + "max", values=np.array([max_val], dtype=np.float32))

    max_out = OGS.Variable(name_prefix + "max_out", dtype=np.float32)
    clip_out = OGS.Variable(name_prefix + "clip_out", dtype=np.float32)

    graph.nodes.extend([
        OGS.Node(op="Max", inputs=[x, min_const], outputs=[max_out], name=name_prefix + "Max"),
        OGS.Node(op="Min", inputs=[max_out, max_const], outputs=[clip_out], name=name_prefix + "Min")
    ])

    # Clip yerine yeni node'ları bağla
    for other_node in graph.nodes:
        for i, inp in enumerate(other_node.inputs):
            if inp.name == y.name:
                other_node.inputs[i] = clip_out

    for i, out in enumerate(graph.outputs):
        if out.name == y.name:
            graph.outputs[i] = clip_out

    node.inputs.clear()
    node.outputs.clear()


#%%
def parse_graph(graph: onnx.ModelProto):
    # Trasverses the computation graph and replaces softmax with fissioned form if applicable
    # Returns a graph with fission applied if applicable
    # Current implementation involves Softmax (with Matmul fusion for the case Fissioned Softmax -> Matmul) and InstanceNorm
    decomposed_op_counters = {}

    # Modified graph
    GS_model = OGS.import_onnx(graph)

    # Read-only graph
    GS_model2 = OGS.import_onnx(graph)

    nodes_list = GS_model.nodes

    for iter in range(len(nodes_list)):
        subgraph_producer = np.array([])
        subgraph_consumer = np.array([])

        # subgraph_producer_matmul = np.array([])
        subgraph_consumer_matmul = np.array([])
        
        node = nodes_list[iter]

        if node.op == "Relu":
            subblock_name = node.name.replace("Relu", "") + str(iter)
            fission_relu(GS_model, node, subblock_name, iter, decomposed_op_counters)

        if node.op == "Dropout":
            subblock_name = node.name.replace("Dropout", "") + str(iter)
            fission_dropout(GS_model, node, subblock_name, iter, decomposed_op_counters)

        if node.op == "Reciprocal":
            subblock_name = node.name.replace("Reciprocal", "") + str(iter)
            fission_reciprocal(GS_model, node, subblock_name, iter, decomposed_op_counters)

        if node.op == "Mish":
            subblock_name = node.name.replace("Mish", "") + str(iter)
            fission_mish(GS_model, node, subblock_name, iter, decomposed_op_counters)

        if node.op == "Clip":
            subblock_name = node.name.replace("Clip", "") + str(iter)
            fission_clip(GS_model, node, subblock_name, iter, decomposed_op_counters)

        if node.op == "Softmax":
            subblock_name = node.name.replace('Softmax', '')
            prev_node = nodes_list[iter-1]
            
            softmax_output_tensor = GS_model2.nodes[iter].outputs[0]
            index_softmax_out = -1
            
            for j in (GS_model2.tensors()).keys():
                # Extract inputs from Softmax Node
                for k in GS_model2.nodes[iter].inputs:
                    if(GS_model2.tensors()[j] == k): subgraph_producer = np.append(subgraph_producer, GS_model2.tensors()[j])
                    
            # Disconnect Softmax node
            node.inputs.clear()
            node.outputs.clear()

            # Extract axis index for Softmax op
            axis_index = range(len((GS_model2.nodes[iter].inputs[0].shape)))[GS_model2.nodes[iter].attrs['axis']]
            
            try:
                next_node = nodes_list[iter+1]
                next_node_inputs = next_node.inputs
                
                for j in (GS_model2.tensors()).keys():
                    for k in range(len(next_node_inputs)):
                        nxt_input = next_node_inputs[k]
                        if(GS_model2.tensors()[j] == nxt_input):
                            if(GS_model2.tensors()[j] != softmax_output_tensor): 
                                subgraph_consumer = np.append(subgraph_consumer, GS_model2.tensors()[j])
                            else:
                                index_softmax_out = k
                            
                matmul_output_tensor = GS_model2.nodes[iter+1].outputs[0]
                index_matmul_out = -1
                
                # If downstream node is Matmul, perform fission + graph transformation, else regular fission
                # Only if with graph transformation, the downstream node of Matmul is required
                next_next_node = None
                
                if next_node.op == 'MatMul':
                    try:
                        next_next_node = nodes_list[iter+2]
                        next_next_node_inputs = next_next_node.inputs
                        
                        for j in (GS_model2.tensors()).keys():
                            for k in range(len(next_next_node_inputs)):
                                nxt_input = next_next_node_inputs[k]
                                if(GS_model2.tensors()[j] == nxt_input):
                                    if(GS_model2.tensors()[j] != matmul_output_tensor): 
                                        subgraph_consumer_matmul = np.append(subgraph_consumer_matmul, GS_model2.tensors()[j])
                                    else:
                                        index_matmul_out = k
                        
                        print(next_next_node.op, " is a valid node; Performing Softmax fission + graph transformation \n")
                        
                    except IndexError:
                        # Matmul is the last node of the graph
                        print("Index Error next_next_node: ", next_node.op, " is the last node; Performing Softmax fission + graph transformation \n")
                        
                    GS_model = fission_softmax_graph_transform(GS_model, GS_model2.nodes[iter].inputs[0], subgraph_consumer, axis_index, subblock_name, decomposed_op_counters)
                    
                else:
                    print("Else Case: ", next_node.op, " is not MatMul; Performing Softmax fission\n")
                    GS_model = fission_softmax2(GS_model, GS_model2.nodes[iter].inputs[0], axis_index, subblock_name, decomposed_op_counters)
                
                # Connect fissioned softmax downstream consumer node
                # fission_div_out tensor name is last node of fissioned graph regardless of possible matmul graph transformation
                # Insertion into consumer depends on whether next node is consumer of softmax or matmul
                if next_next_node is None:
                    assert(index_softmax_out >= 0)
                    
                    if(len(subgraph_consumer) > 0):
                        subgraph_consumer = np.insert(subgraph_consumer, index_softmax_out, GS_model.tensors()[subblock_name + 'fission_div_out'])
                    else:
                        subgraph_consumer = np.append(subgraph_consumer, GS_model.tensors()[subblock_name + 'fission_div_out'])
                    
                    next_node.inputs = subgraph_consumer
                else:
                    assert(index_matmul_out >= 0)
                    
                    if(len(subgraph_consumer_matmul) > 0):
                        subgraph_consumer_matmul = np.insert(subgraph_consumer_matmul, index_matmul_out, GS_model.tensors()[subblock_name + 'fission_div_out'])
                    else:
                        subgraph_consumer_matmul = np.append(subgraph_consumer_matmul, GS_model.tensors()[subblock_name + 'fission_div_out'])
                    
                    next_next_node.inputs = subgraph_consumer_matmul
                
            except IndexError:
                # Valid to perform fission on softmax only
                # next_node = None, subgraph_consumer_softmax = []
                print(node.op, " is the last node; Performing fission\n")
                
                next_node = None
                GS_model = fission_softmax2(GS_model, GS_model2.nodes[iter].inputs[0], axis_index, subblock_name, decomposed_op_counters)

                # If softmax is the output node
                subgraph_consumer = np.append(subgraph_consumer, GS_model.tensors()[subblock_name + 'fission_div_out'])
                
                # Set graph output to softmax.output
                GS_model.outputs = [GS_model.tensors()[subblock_name + 'fission_div_out']]
            
            # Connect parent node to fissioned softmax input
            prev_node.outputs = [GS_model.tensors()[subblock_name + 'prev_out']]
        
        if node.op == "InstanceNormalization":
            subblock_name = node.name.replace('InstanceNormalization', '') + str(iter)
            prev_node = nodes_list[iter-1]
            print("Ins norm subblock name", subblock_name)
            
            instnorm_output_tensor = GS_model2.nodes[iter].outputs[0]
            index_instnorm_out = -1
            
            for j in (GS_model2.tensors()).keys():
                # Extract inputs from Softmax Node
                for k in GS_model2.nodes[iter].inputs:
                    if(GS_model2.tensors()[j] == k):
                        subgraph_producer = np.append(subgraph_producer, GS_model2.tensors()[j])
                    
            

            GS_model = fission_instancenorm(GS_model, GS_model2.nodes[iter].inputs[0],
                                            GS_model2.nodes[iter].inputs[1], GS_model2.nodes[iter].inputs[2],
                                            GS_model2.nodes[iter].attrs['epsilon'],
                                            subblock_name, decomposed_op_counters)
            
            # Disconnect Softmax node
            node.inputs.clear()
            node.outputs.clear()
            
            # Connect parent node to fissioned softmax input
            # prev_node.outputs = [GS_model.tensors()[subblock_name + 'x']]

            try:
                print(node.op, " is not the last node; Performing fission\n")
                next_node = nodes_list[iter+1]
                #TODO: Connect all/multiple next nodes in the graph
                next_node_inputs = next_node.inputs
                
                for j in (GS_model2.tensors()).keys():
                    for k in range(len(next_node_inputs)):
                        nxt_input = next_node_inputs[k]
                        if(GS_model2.tensors()[j] == nxt_input):
                            if(GS_model2.tensors()[j] != instnorm_output_tensor): 
                                subgraph_consumer = np.append(subgraph_consumer, GS_model2.tensors()[j])
                            else:
                                index_instnorm_out = k

                if(len(subgraph_consumer) > 0 and next_node is not None):
                    subgraph_consumer = np.insert(subgraph_consumer, index_instnorm_out, GS_model.tensors()[subblock_name + '_insnorm_fission_x_norm_shift_out'])
                else:
                    subgraph_consumer = np.append(subgraph_consumer, GS_model.tensors()[subblock_name + '_insnorm_fission_x_norm_shift_out'])
            
                next_node.inputs = subgraph_consumer

            except IndexError:
                # Valid to perform fission on softmax only
                # next_node = None, subgraph_consumer_softmax = []
                print(node.op, " is the last node; Performing fission\n")
                
                GS_model = fission_instancenorm(GS_model, GS_model2.inputs[0],
                                            GS_model2.inputs[1], GS_model2.inputs[2],
                                            GS_model2.nodes[iter].attrs['epsilon'].values[0],
                                            subblock_name, decomposed_op_counters)

                next_node = None
                # Set graph output to softmax.output
                GS_model.outputs = [GS_model.tensors()[subblock_name + '_insnorm_fission_x_norm_shift_out']]
        if node.op == "LayerNormalization":
            subblock_name = node.name.replace("LayerNormalization", "") + str(iter)
            prev_node = nodes_list[iter - 1] if iter > 0 else None

            ln_output_tensor = GS_model2.nodes[iter].outputs[0]
            index_ln_out = -1
            subgraph_consumer = np.array([])

            x, scale, bias = GS_model2.nodes[iter].inputs
            epsilon = node.attrs.get("epsilon", 1e-5)

            # Fission işlemi
            output_var = fission_layernorm(GS_model, x, scale, bias, epsilon, subblock_name, node,decomposed_op_counters)

            ln_output_tensor = GS_model2.nodes[iter].outputs[0]

            # Tüm node'larda eski output'u input olarak kullananları güncelle
            for other_node in nodes_list:
                for i, input_tensor in enumerate(other_node.inputs):
                    if input_tensor.name == ln_output_tensor.name:  
                        other_node.inputs[i] = output_var

            node.inputs.clear()
            node.outputs.clear()

            try:
                next_node = nodes_list[iter + 1]
                next_node_inputs = next_node.inputs

                for j in GS_model2.tensors().keys():
                    for k in range(len(next_node_inputs)):
                        nxt_input = next_node_inputs[k]
                        if GS_model2.tensors()[j] == nxt_input:
                            if GS_model2.tensors()[j] == ln_output_tensor:
                                index_ln_out = k
                            else:
                                subgraph_consumer = np.append(subgraph_consumer, GS_model2.tensors()[j])
                
                if len(subgraph_consumer) > 0:
                    subgraph_consumer = np.insert(subgraph_consumer, index_ln_out, output_var)
                else:
                    subgraph_consumer = np.append(subgraph_consumer, output_var)

                next_node.inputs = subgraph_consumer

            except IndexError:
                print("LayerNorm is the last node")
                # Replace model output if matched
                for i, output in enumerate(GS_model.outputs):
                    if output.name == ln_output_tensor.name:
                        GS_model.outputs[i] = shifted
                        break
                else:
                    # If original output is not found, append it
                    GS_model.outputs.append(shifted)

        if node.op == "HardSigmoid":
            subblock_name = node.name.replace("HardSigmoid", "") + str(iter)
            fission_hardsigmoid(GS_model, node, subblock_name, iter, decomposed_op_counters)

        if node.op == "HardSwish":
            subblock_name = node.name.replace("HardSwish", "") + str(iter)
            fission_hardswish(GS_model, node, subblock_name, iter, decomposed_op_counters)

        if node.op == "Sigmoid":
            subblock_name = node.name.replace("Sigmoid", "") + str(iter)
            fission_sigmoid(GS_model, node, subblock_name, iter, decomposed_op_counters)

        if node.op == "Tanh":
            subblock_name = node.name.replace("Tanh", "") + str(iter)
            fission_tanh(GS_model, node, subblock_name, iter, decomposed_op_counters)
        if node.op == "LeakyRelu":
            subblock_name = node.name.replace("LeakyRelu", "") + str(iter)
            fission_leaky_relu(GS_model, node, subblock_name, iter, decomposed_op_counters)

        if node.op == "Split":
            # Change split to equivalent Slices
            input_tensor = node.inputs[0]
            output_tensors = node.outputs
            axis = node.attrs["axis"]
            if "split" in node.attrs:
                split_sizes = node.attrs["split"]
            else:
                split_sizes = node.inputs[1].values
            start_index = 0
            for output_tensor, split_size in zip(output_tensors, split_sizes):
                axes = OGS.Constant(output_tensor.name + "axes", values=np.array([axis]))
                starts = OGS.Constant(output_tensor.name + "starts", values=np.array([start_index]))
                ends = OGS.Constant(output_tensor.name + "ends", values=np.array([start_index + split_size]))
                slice_node = OGS.Node(
                    op="Slice",
                    inputs=[input_tensor, starts, ends, axes],
                    outputs=[output_tensor]
                )
                GS_model.nodes.append(slice_node)
                start_index += split_size
            # Disconnect Split node
            node.inputs.clear()
            node.outputs.clear()

    try:
        GS_model.cleanup().toposort()
        
        produced_outputs = {tensor.name for node in GS_model.nodes for tensor in node.outputs}
        GS_model.outputs = [o for o in GS_model.outputs if o.name in produced_outputs]

        fisson_onnx_graph = OGS.export_onnx(GS_model)


        
    except:
        # For some reason graph is correct but clean-up fails
        # Workaround is to save as onnx, load in again and then do cleanup
        onnx.save(OGS.export_onnx(GS_model), "tmp.onnx")
        reload_onnx = onnx.load("tmp.onnx")
        os.remove("tmp.onnx")

        reload_GS_model = OGS.import_onnx(reload_onnx)
        reload_GS_model.cleanup().toposort()

        produced_outputs = {tensor.name for node in reload_GS_model.nodes for tensor in node.outputs}
        reload_GS_model.outputs = [o for o in reload_GS_model.outputs if o.name in produced_outputs]

        fisson_onnx_graph = OGS.export_onnx(reload_GS_model)


        
        
    fisson_onnx_graph_shape_inferred = shape_inference.infer_shapes(fisson_onnx_graph)
    
    return fisson_onnx_graph_shape_inferred, decomposed_op_counters  #fisson_onnx_graph_shape_inferred
            
def fission_softmax_graph_transform(graph, a, matmul_inputs, softmax_axis, subblock_name, counter_dict):
    counter_dict["Softmax"] = counter_dict.get("Softmax", 0) + 1
    # Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
    # requires normalized input to Exp -> Fine max() along each row dim
    # Enter here if downstream node is confirmed to be matmul
    # Perform graph transformation to promote matmul and do fission
    # matmul_inputs will contain all inputs other than the softmax from the pre-existing graph
    
    prev_out = OGS.Variable(subblock_name + "prev_out", dtype=np.float32)
    # max_out = OGS.Variable(subblock_name + "max_out", dtype=np.float32)
    # sub_out = OGS.Variable(subblock_name + "sub_out", dtype=np.float32)
    exp_out = OGS.Variable(subblock_name + "exp_out", dtype=np.float32)
    concat_out = OGS.Variable(subblock_name + "concat_out", dtype=np.float32)
    fused_matmul_out = OGS.Variable(subblock_name + "fused_matmul_out", dtype=np.float32)
    slice_out_0 = OGS.Variable(subblock_name + "slice_out_0", dtype=np.float32)
    slice_out_1 = OGS.Variable(subblock_name + "slice_out_1", dtype=np.float32)
    fission_div_out = OGS.Variable(subblock_name + "fission_div_out", dtype=np.float32)
    
    c2 = OGS.Node(
            op= "Exp",
            # inputs=[sub_out],
            inputs=[prev_out],
            outputs=[exp_out],
            name=subblock_name + "Fission_Exp"
        )

    graph.nodes.append(c2)

    # Implement ReduceSum as GEMV
    # Impl ReduceSum as Matrix * Vector multiply
    # tuple(np.append(a.shape[:-1], 1))
    
    tensor_shape = list(a.shape)
    tensor_shape.pop(-2)
    tensor_shape = tuple(np.append(tensor_shape, 1))
    ones_vector = OGS.Constant(subblock_name + "ones_vector", values=np.ones(shape=tensor_shape, dtype=np.float32))
    
    # Concatenate the ones_vector along with the input to downstream matmul (not from softmax)
    # TODO: Change Cancat to Pad
    
    c3 = OGS.Node(
        op="Concat",
        inputs=np.append(matmul_inputs, ones_vector),
        outputs=[concat_out],
        attrs={'axis': softmax_axis},
        name=subblock_name + "Concat_Matmul_Inputs"
    )
    
    graph.nodes.append(c3)
    
    # Define the Matmul as fused multiply with concatenated nodes
    c4 = OGS.Node(
            op= "MatMul",
            inputs=[exp_out, concat_out],
            outputs=[fused_matmul_out],
            name=subblock_name + "Fused_MatMul"
        )
    
    graph.nodes.append(c4)
    
    # Define the split node to recover tensors for division op
    
    concat_tensor_shape = list(a.shape)
    concat_tensor_shape.pop(-1)
    # Adding one to indicate concat dimension
    concat_tensor_shape = np.append(concat_tensor_shape, matmul_inputs[0].shape[-1] + 1)
    
    np_starts_0 = np.zeros(shape=len(concat_tensor_shape), dtype=np.int64)
    starts_0 = OGS.Constant(subblock_name + "starts_0", values=np_starts_0)
    # starts_0 = OGS.Constant("starts_0", values=np.array([0, 0, 0, 0]))
    
    np_ends_0 = copy.deepcopy(concat_tensor_shape)
    np_ends_0[-1] -= 1
    
    ends_0 = OGS.Constant(subblock_name + "ends_0", values=np_ends_0)
    # ends_0 = OGS.Constant("ends_0", values=np.array([1, 1, 65536, 32]))
    
    c5 = OGS.Node(
        op="Slice",
        inputs=[fused_matmul_out, starts_0, ends_0],
        outputs=[slice_out_0],
        name=subblock_name + "Slice_MatMul_Output_Num"
    )

    graph.nodes.append(c5)
    
    np_starts_1 = np.zeros(shape=len(concat_tensor_shape), dtype=np.int64)
    np_starts_1[-1] = np_ends_0[-1]
    starts_1 = OGS.Constant(subblock_name + "starts_1", values=np_starts_1)
    
    np_ends_1 = copy.deepcopy(concat_tensor_shape)
    ends_1 = OGS.Constant(subblock_name + "ends_1", values=np_ends_1)
    # starts_1 = OGS.Constant("starts_1", values=np.array([0, 0, 0, 32]))
    # ends_1 = OGS.Constant("ends_1", values=np.array([1, 1, 65536, 33]))
    
    c6 = OGS.Node(
        op="Slice",
        inputs=[fused_matmul_out, starts_1, ends_1],
        outputs=[slice_out_1],
        name=subblock_name + "Slice_MatMul_Output_Denom"
    )

    graph.nodes.append(c6)
    
    # onnx and tvm.relay support implicit numpy style broadcast
    
    c7 = OGS.Node(
            op = "Div",
            inputs=[slice_out_0, slice_out_1],
            outputs=[fission_div_out],
            name=subblock_name + "Fission_Div"
        )
    
    graph.nodes.append(c7)
    
    return graph


#%%
def fission_softmax2(graph, a, softmax_axis, subblock_name, counter_dict):
    counter_dict["Softmax"] = counter_dict.get("Softmax", 0) + 1
    # Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
    # requires normalized input to Exp -> Fine max() along each row dim
    prev_out = OGS.Variable(subblock_name + "prev_out", dtype=np.float32)
    # max_out = OGS.Variable(subblock_name + "max_out", dtype=np.float32)
    # sub_out = OGS.Variable(subblock_name + "sub_out", dtype=np.float32)
    exp_out = OGS.Variable(subblock_name + "exp_out", dtype=np.float32)
    reduced_sum_out = OGS.Variable(subblock_name + "reduced_sum_out", dtype=np.float32)
    fission_div_out = OGS.Variable(subblock_name + "fission_div_out", dtype=np.float32)

    c2 = OGS.Node(
            op= "Exp",
            # inputs=[sub_out],
            inputs=[prev_out],
            outputs=[exp_out],
            name=subblock_name + "Fission_Exp"
        )

    graph.nodes.append(c2)

    # Implement ReduceSum as GEMV
    if softmax_axis != -1 or softmax_axis != a.shape[-1]:
        # Impl ReduceSum as Matrix * Vector multiply
        tensor_shape = list(a.shape)
        tensor_shape.pop(-2)
        tensor_shape = tuple(np.append(tensor_shape, 1))
        ones_vector = OGS.Constant(subblock_name + "ones_vector", values=np.ones(shape=tensor_shape, dtype=np.float32))
        
        c3 = OGS.Node(
                op= "MatMul",
                inputs=[exp_out, ones_vector],
                outputs=[reduced_sum_out],
                name=subblock_name + "Fission_MatMul"
            )
        
        graph.nodes.append(c3)
        
    else:
        # Requires flattening + matmul + reshape to recover original tensor
        print("softmax for non-innermost dim is not implemented")
    
    # onnx and tvm.relay support implicit numpy style broadcast
    
    c4 = OGS.Node(
            op = "Div",
            inputs=[exp_out, reduced_sum_out],
            outputs=[fission_div_out],
            name=subblock_name + "Fission_Div"
        )
    
    graph.nodes.append(c4)
    
    return graph
    
    
#%%
def fission_softmax(a, softmax_axis, counter_dict):
    counter_dict["Softmax"] = counter_dict.get("Softmax", 0) + 1
    # Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
    # requires normalized input to Exp -> Fine max() along each row dim
    input = OGS.Variable("input", shape=a.shape, dtype=np.float32)
    max_out = OGS.Variable("max_out", dtype=np.float32)
    sub_out = OGS.Variable("sub_out", dtype=np.float32)
    exp_out = OGS.Variable("exp_out", dtype=np.float32)
    reduced_sum_out = OGS.Variable("reduced_sum_out", dtype=np.float32)
    
    output = OGS.Variable("output", shape=a.shape, dtype=np.float32)
    
    c0 = OGS.Node(
            op="ReduceMax",
            inputs=[input],
            outputs=[max_out],
            attrs={'axes': [softmax_axis], 'keepdims': 1},
            name="ReduceMax"
        )
    
    c1 = OGS.Node(
            op="Sub",
            inputs=[input, max_out],
            outputs=[sub_out],
            name="Sub"
        )
    
    c2 = OGS.Node(
            op= "Exp",
            inputs=[sub_out],
            outputs=[exp_out],
            name="Exp"
        )
    
    # Implement ReduceSum as GEMV
    if softmax_axis != -1 or softmax_axis != a.shape[-1]:
        # Impl ReduceSum as Matrix * Vector multiply
        tensor_shape = list(a.shape)
        tensor_shape.pop(-2)
        tensor_shape = tuple(np.append(tensor_shape, 1))
        
        ones_vector = OGS.Constant("ones_vector", values=np.ones(shape=tensor_shape, dtype=np.float32))
        
        c3 = OGS.Node(
                op= "MatMul",
                inputs=[exp_out, ones_vector],
                outputs=[reduced_sum_out],
                name="MatMul"
            )
    else:
        # Requires flattening + matmul + reshape to recover original tensor
        print("softmax for non-innermost dim is not implemented")
    
    # onnx and tvm.relay support implicit numpy style broadcast
    c4 = OGS.Node(
            op = "Div",
            inputs=[exp_out, reduced_sum_out],
            outputs=[output],
            name="Div"
        )
    
    fissioned_softmax = OGS.Graph(nodes=[c0, c1, c2, c3, c4], inputs=[input], outputs=[output])
    return fissioned_softmax



def fission_instancenorm(graph, x, s, bias, epsilon, subblock_name,counter_dict):
    counter_dict["InstanceNormalization"] = counter_dict.get("InstanceNormalization", 0) + 1
    #ref https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/nn/instance_norm.cc
    #ref https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/instancenorm.py

    # x: BxCxHxW
    # Normalization across HxW

    dims_x = len(x.shape)
    mean_axes = range(2, dims_x)
    # GS_model2 = OGS.import_onnx(graph)

    nodes_list = []


    name_prefix = subblock_name + "_insnorm_fission_"

    # x = OGS.Variable("x", dtype=np.float32)
    mean_out = OGS.Variable(name_prefix+"mean_out", dtype=np.float32)
    var_out = OGS.Variable(name_prefix+"var_out", dtype=np.float32)
    x_minus_mean = OGS.Variable(name_prefix+"x_minus_mean", dtype=np.float32)
    x_minus_mean_squared = OGS.Variable(name_prefix+"x_minus_mean_squared", dtype=np.float32)
    eps_val = OGS.Constant(name_prefix+"eps_val", values=np.array([epsilon], dtype=np.float32))
    # scale_val = OGS.Constant("scale_val", values=np.array([s]))
    # shift_val = OGS.Constant("shift_val", values=np.array([bias]))
    # eps_val = OGS.Variable("eps_val", dtype=np.float32).to_constant(np.array([epsilon], dtype=np.float32))
    scale_val = s
    shift_val = bias

    dim_ones = (1,) * (dims_x - 2)
    
    # ones_vector = OGS.Constant("ones_vector", values=np.ones(shape=dim_ones, dtype=np.float32))
    s.shape.extend(dim_ones)
    bias.shape.extend(dim_ones)

    var_plus_eps = OGS.Variable(name_prefix+"var_plus_eps", dtype=np.float32)
    sqrt_var_plus_eps = OGS.Variable(name_prefix+"sqrt_var_plus_eps", dtype=np.float32)
    x_norm = OGS.Variable(name_prefix+"x_norm", dtype=np.float32)

    x_norm_scale_out = OGS.Variable(name_prefix+"x_norm_scale_out", dtype=np.float32)
    x_norm_shift_out = OGS.Variable(name_prefix+"x_norm_shift_out", dtype=np.float32)


    c0 = OGS.Node(
            op="ReduceMean",
            inputs=[x],
            outputs=[mean_out],
            attrs={'axes': mean_axes, 'keepdims': 1},
            name=name_prefix + "_ReduceMean"
        )


    nodes_list.append(c0)

    c1 = OGS.Node(
            op="Sub",
            inputs=np.append([x], mean_out),
            outputs=[x_minus_mean],
            name=name_prefix + "_Subx_minus_mean"
        )
    nodes_list.append(c1)


    c2 = OGS.Node(
            op="Mul",
            inputs=np.append([x_minus_mean], x_minus_mean),
            outputs=[x_minus_mean_squared],
            name=name_prefix + "_Mul_x_minus_mean_squared"
        )

    nodes_list.append(c2)
    
    c3 = OGS.Node(
            op="ReduceMean",
            inputs=[x_minus_mean_squared],
            outputs=[var_out],
            attrs={'axes': mean_axes, 'keepdims': 1},
            name=name_prefix + "_ReduceMean_var"
        )
    nodes_list.append(c3)


    c4 = OGS.Node(
            op = "Sum",
            inputs=[var_out, eps_val],
            outputs=[var_plus_eps],
            name=name_prefix+"_sum_var_eps"
        )

    nodes_list.append(c4)

    c5 = OGS.Node(
            op="Sqrt",
            inputs=[var_plus_eps],
            outputs=[sqrt_var_plus_eps],
            name=name_prefix + "_Sqrt_var_eps"
        )

    nodes_list.append(c5)

    c6 = OGS.Node(
            op="Div",
            inputs=[x_minus_mean, sqrt_var_plus_eps],
            outputs=[x_norm],
            name=name_prefix + "_Div_x_norm"
        )

    nodes_list.append(c6)

    c8 = OGS.Node(
            op="Mul",
            inputs=[x_norm, s],
            outputs=[x_norm_scale_out],
            name=name_prefix + "_Mul_x_norm_scale"
        )

    nodes_list.append(c8)

    c10 = OGS.Node(
            op="Sum",
            inputs=[x_norm_scale_out, bias],
            outputs=[x_norm_shift_out],
            name=name_prefix + "_Sum_x_norm_shift"
        )

    nodes_list.append(c10)

    # OGS_graph = OGS.Graph(nodes=nodes_list, inputs=[x, scale_val, shift_val], outputs=[x_norm_shift_out])
    # onnx_graph = OGS.export_onnx(OGS_graph)
    # onnx_graph_shape_inferred = shape_inference.infer_shapes(onnx_graph)
    
    for n in nodes_list:
        graph.nodes.append(n)
    
    return graph




def perform_op_fission(graph: onnx.ModelProto):
    # Top-level entry function to parse the graph and return fission-ed graph if possible
    graph_si = shape_inference.infer_shapes(graph)
    fission_graph, counters = parse_graph(graph_si)
    
    return fission_graph, counters
    

if __name__=="__main__":
        
    onnx_model = onnx.load(sys.argv[1])

    shape_dict, input_name, input_shape, DTYPE = get_input_node_info(onnx_model)
    
    fission_testcase_onnx, decomposed_op_counters = perform_op_fission(onnx_model)
    
    GS_model = OGS.import_onnx(fission_testcase_onnx)
    
    onnx.save(fission_testcase_onnx, sys.argv[2])
    print("\n=== Decomposition Summary ===")
    for op_type, count in decomposed_op_counters.items():
        print(f"{op_type}: {count} node(s) decomposed")