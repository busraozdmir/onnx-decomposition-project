import argparse
import onnx
import onnxruntime as ort
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_decomposition_pipeline(model_info, log_file):
    log = []
    log.append(f"\n Testing: {model_info['name']}")
    print(log[-1])

    print("Modified model path:", model_info["modified_model_path"])


    original_sess = ort.InferenceSession(model_info["original_model_path"], providers=["CUDAExecutionProvider"])
    modified_sess = ort.InferenceSession(model_info["modified_model_path"], providers=["CUDAExecutionProvider"])


    for input_data in model_info["input_data_list"]:
        orig_out = original_sess.run(None, input_data)
        mod_out = modified_sess.run(None, input_data)
        assert all(np.allclose(o, m, atol=1e-2) for o, m in zip(orig_out, mod_out)), " Output mismatch!"

    log.append(" Output consistency test passed.")
    print(log[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name to test")
    args = parser.parse_args()

    np.random.seed(42)

    all_models = {
        "mobilenetv3": {
            "original_model_path": "input_models/mobilenetv3_small_075_Opset17.onnx",
            "modified_model_path": "output_models/modified_mobilenetv3.onnx",
            "input_data_list": [{"x": np.random.rand(1, 3, 224, 224).astype(np.float32)} for _ in range(50)]
        },
        "tinyyolov2": {
            "original_model_path": "input_models/tinyyolov2-8.onnx",
            "modified_model_path": "output_models/modified_tinyyolov2.onnx",
            "input_data_list": [{"image": np.random.rand(1, 3, 416, 416).astype(np.float32)} for _ in range(50)]
        },
        "bert": {
            "original_model_path": "input_models/bert_Opset17.onnx",
            "modified_model_path": "output_models/modified_bert.onnx",
            "input_data_list": [
                {"input_ids": np.random.randint(0, 100, (1, 128)).astype(np.int64),
                 "attention_mask": np.random.rand(1, 128).astype(np.float32)}
                for _ in range(50)
            ]
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
        "roberta": {
            "original_model_path": "input_models/roberta_Opset18.onnx",
            "modified_model_path": "output_models/modified_roberta.onnx",
            "input_data_list": [
                {"input_ids": np.random.randint(0, 100, (1, 128)).astype(np.int64),
                 "attention_mask": np.random.rand(1, 128).astype(np.float32)}
                for _ in range(50)
            ]
        },
        "adv_inception": {
            "original_model_path": "input_models/adv_inception_v3_Opset18.onnx",
            "modified_model_path": "output_models/modified_adv_inception.onnx",
            "input_data_list": [{"x": np.random.rand(1, 3, 299, 299).astype(np.float32)} for _ in range(50)]
        },
        "densenet201": {
            "original_model_path": "input_models/densenet201_Opset18.onnx",
            "modified_model_path": "output_models/modified_densenet.onnx",
            "input_data_list": [{"x": np.random.rand(1, 3, 224, 224).astype(np.float32)} for _ in range(50)]
        },
        "resnet50_v1_12": {
            "original_model_path": "input_models/resnet50-v1-12.onnx",
            "modified_model_path": "output_models/modified_resnet50_v1_12.onnx",
            "input_data_list": [{"data": np.random.rand(1, 3, 224, 224).astype(np.float32)} for _ in range(50)]
        },
        "mobilenet_v1": {
            "original_model_path": "input_models/mobilenet_v1_1.0_224_opset13.onnx",
            "modified_model_path": "output_models/modified_mobilenet_v1.onnx",
            "input_data_list": [{"input:0": np.random.rand(1, 3, 224, 224).astype(np.float32)} for _ in range(50)]
        },
        "layernorm_opset17": {
            "original_model_path": "input_models/single_ln_mem_bound_model.onnx",
            "modified_model_path": "output_models/modified_layernorm_test.onnx",
            "input_data_list": [{"input": np.random.rand(1, 16).astype(np.float32)} for _ in range(50)]
        },
        "dinov2_vits14": {
            "original_model_path": "input_models/dinov2_vits14.onnx",
            "modified_model_path": "output_models/modified_dinov2_vits14.onnx",
            "input_data_list": [{"pixel_values": np.random.rand(1, 3, 224, 224).astype(np.float32)} for _ in range(50)]
        },
        "swin_s3": {
            "original_model_path": "input_models/swin_s3_small_224_Opset17.onnx",
            "modified_model_path": "output_models/modified_swin_s3.onnx",
            "input_data_list": [{"data": np.random.rand(1, 3, 224, 224).astype(np.float32)} for _ in range(50)]
        },
        "convnext_small_opset17":{
            "original_model_path": "input_models/convnext_small_Opset17.onnx",
            "modified_model_path": "output_models/modified_convnext.onnx",
            "input_data_list": [{"x": np.random.rand(1, 3, 224, 224).astype(np.float32)} for _ in range(50)]
        },
        "segformer":{
            "original_model_path": "input_models/segformer.onnx",
            "modified_model_path": "output_models/decomposed_segformer.onnx",
            "input_data_list": [{"input.1": np.random.rand(1, 3, 512, 512).astype(np.float32)} for _ in range(50)]
        },
        "candy":{
            "original_model_path": "input_models/candy.onnx",
            "modified_model_path": "output_models/fused_candy.onnx",
            "input_data_list": [{"input1": np.random.rand(1, 3, 224, 224).astype(np.float32)} for _ in range(50)]
        },
        "yolov4":{
            "original_model_path": "input_models/yolov4.onnx",
            "modified_model_path": "output_models/modified_yolov4.onnx",
            "input_data_list": [{"input.1": np.random.rand(1, 3, 416, 416).astype(np.float32)} for _ in range(50)]
        },
        "yolox":{
            "original_model_path": "input_models/yolox.onnx",
            "modified_model_path": "output_models/modified_yolox.onnx",
            "input_data_list": [{"onnx::Slice_0": np.random.rand(1, 3, 416, 416).astype(np.float32)} for _ in range(50)]
        }
    }

    model_key = args.model.lower()
    if model_key not in all_models:
        print("\n Invalid model name. Choose from:")
        for key in all_models:
            print(" -", key)
        exit(1)

    model_info = all_models[model_key]
    model_info["name"] = model_key
    log_file = f"decomposition_results_{model_key}.txt"
    with open(log_file, "w") as f:
        f.write("== DECOMPOSITION TEST RESULTS ==\n")

    test_decomposition_pipeline(model_info, log_file)
