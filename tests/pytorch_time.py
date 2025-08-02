import argparse
import numpy as np
import torch
import onnx
from onnx2torch import convert
import time
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

np.random.seed(42)

all_models = {
    "mobilenetv3": {
        "original_model_path": "input_models/mobilenetv3_small_075_Opset17.onnx",
        "modified_model_path": "output_models/modified_mobilenetv3.onnx",
        "input_data_list": [{"x": np.random.rand(1, 3, 224, 224).astype(np.float32)} for _ in range(50)]
    },
    "efficientnet": {
        "original_model_path": "input_models/efficientnet_b0_Opset17.onnx",
        "modified_model_path": "output_models/modified_efficientnet.onnx",
        "input_data_list": [{"x": np.random.rand(1, 3, 224, 224).astype(np.float32)} for _ in range(50)]
    },
    "resnet50": {
        "original_model_path": "input_models/resnet50_Opset18.onnx",
        "modified_model_path": "output_models/modified_resnet50.onnx",
        "input_data_list": [{"x": np.random.rand(1, 3, 224, 224).astype(np.float32)} for _ in range(50)]
    },
    "resnet50_v1_12": {
        "original_model_path": "input_models/resnet50-v1-12.onnx",
        "modified_model_path": "output_models/modified_resnet50_v1_12.onnx",
        "input_data_list": [{"data": np.random.rand(1, 3, 224, 224).astype(np.float32)} for _ in range(50)]
    },
    "segformer": {
        "original_model_path": "input_models/segformer.onnx",
        "modified_model_path": "output_models/decomposed_segformer.onnx",
        "input_data_list": [{"input.1": np.random.rand(1, 3, 512, 512).astype(np.float32)} for _ in range(50)]
    },
    "candy": {
        "original_model_path": "input_models/candy.onnx",
        "modified_model_path": "output_models/fused_candy.onnx",
        "input_data_list": [{"input1": np.random.rand(1, 3, 224, 224).astype(np.float32)} for _ in range(50)]
    },
    "yolov4": {
        "original_model_path": "input_models/yolov4.onnx",
        "modified_model_path": "output_models/modified_yolov4.onnx",
        "input_data_list": [{"input.1": np.random.rand(1, 3, 416, 416).astype(np.float32)} for _ in range(50)]
    },
    "yolox": {
        "original_model_path": "input_models/yolox.onnx",
        "modified_model_path": "output_models/modified_yolox.onnx",
        "input_data_list": [{"onnx::Slice_0": np.random.rand(1, 3, 416, 416).astype(np.float32)} for _ in range(50)]
    },
    "efficientvit": {
        "original_model_path": "input_models/efficientvit_cls_l1_r224.onnx",
        "modified_model_path": "output_models/modified_efficientvit.onnx",
        "input_data_list": [{"input.1": np.random.rand(1, 3, 224, 224).astype(np.float32)} for _ in range(50)]
    }
}

def run_inference(model_name: str, mode: str): 
    model_entry = all_models[model_name]
    model_path = model_entry[f"{mode}_model_path"]
    input_list = model_entry["input_data_list"]

    print(f"Loading {mode} model for {model_name}: {model_path}")
    onnx_model = onnx.load(model_path)
    torch_model = convert(onnx_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_model = torch_model.to(device)
    compiled_model = torch.compile(torch_model)

    compiled_model.eval()
    with torch.no_grad():
        # Warm-up
        for i in range(5):
            inputs = [torch.from_numpy(v).to(device) for v in input_list[i].values()]
            compiled_model(*inputs)

        # Inference timing
        times = []
        for input_dict in input_list:
            torch_inputs = [torch.from_numpy(v).to(device) for v in input_dict.values()]
            start = time.time()
            compiled_model(*torch_inputs)
            end = time.time()
            times.append(end - start)


    avg_time = sum(times) / len(times)
    print(f"Avg inference time for {model_name} ({mode}): {avg_time:.6f} seconds")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=all_models.keys(), help="Model name to test")
    parser.add_argument("--mode", type=str, required=True, choices=["original", "modified"], help="Model version")
    args = parser.parse_args()

    run_inference(args.model, args.mode)
