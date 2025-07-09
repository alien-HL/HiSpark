import argparse
import numpy as np
from plateNet import myNet_ocr
from alphabets import plate_chr
import torch
import onnx

def remove_headpose_squeeze_node(onnx_path, save_path):
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph

    ## 添加常数 input
    shape = onnx.helper.make_tensor('shape', onnx.TensorProto.INT64, [3], [1,78,21])
    graph.initializer.append(shape)
    for i in range(len(graph.node)):
        if graph.node[i].op_type == "Squeeze":
            reshape_node_def = onnx.helper.make_node(
                        'Reshape', # node name
                        inputs=[graph.node[i].input[0], 'shape'], # inputs
                        outputs=[graph.node[i].output[0]], # outputs
                        name = graph.node[i].name
                    )
            graph.node.remove(graph.node[i])
            graph.node.insert(i, reshape_node_def)

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, save_path)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/home/nicta100-s26/ai/sample/ocr/crnn_plate_recognition/saved_model/best.pth', help='weights path')  # from yolov5/models/
    parser.add_argument('--save_path', type=str, default='/home/nicta100-s26/ai/sample/ocr/crnn_plate_recognition/saved_model/best.onnx', help='onnx save path')
    parser.add_argument('--img_size', nargs='+', type=int, default=[48, 168], help='image size')  # height, width
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', default=False, help='enable dynamic axis in onnx model')
    parser.add_argument('--simplify', action='store_true', default=True, help='simplified onnx')
    # parser.add_argument('--trt', action='store_true', default=False, help='support trt')



    opt = parser.parse_args()
    print(opt)
    checkpoint = torch.load(opt.weights)
    cfg = checkpoint['cfg']
    model = myNet_ocr(num_classes=len(plate_chr),cfg=cfg,export=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    input = torch.randn(opt.batch_size,3,48,168)
    onnx_file_name = opt.save_path

    torch.onnx.export(model,input,onnx_file_name,
                      input_names=["images"],output_names=["output"],
                      verbose=False,
                      opset_version=10,
                      dynamic_axes={'images': {0: 'batch'},
                                    'output': {0: 'batch'}
                                   } if opt.dynamic else None)
    print(f"convert completed,save to {opt.save_path}")
    if opt.simplify:
        from onnxsim import simplify
        print(f"begin simplify ....")
        input_shapes = {"images": list(input.shape)}
        onnx_model = onnx.load(onnx_file_name)
        model_simp, check = simplify(onnx_model,test_input_shapes=input_shapes)
        onnx.save(model_simp, onnx_file_name)
        print(f"simplify completed,save to {opt.save_path}")

    remove_headpose_squeeze_node(opt.save_path,opt.save_path)
