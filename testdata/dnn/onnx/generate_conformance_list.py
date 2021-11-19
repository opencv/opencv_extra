import cv2
import os

MODEL_NAME = 'model.onnx'
DATA_DIR = 'test_data_set_0'

TEST_DATA_PATH = os.environ['OPENCV_TEST_DATA_PATH']

root_dir = os.path.join(TEST_DATA_PATH, 'dnn/onnx/conformance')
node_dir = os.path.join(TEST_DATA_PATH, 'dnn/onnx/conformance/node')

print('{')
for test_name in os.listdir(node_dir):
    test_path = os.path.join(node_dir, test_name)

    assert os.path.isdir(test_path), 'node folder should contain only directories'
    children = sorted([x for x in os.listdir(test_path)])
    assert children == [MODEL_NAME, DATA_DIR], 'test folder should contain model and one dataset'

    data_prefix = os.path.join(DATA_DIR)
    dataset_path = os.path.join(test_path, DATA_DIR)

    inputs = 0
    outputs = 0
    for data_name in os.listdir(dataset_path):
        data_path = os.path.join(data_prefix, data_name)
        if data_name.startswith('input_'):
            inputs += 1
        else:
            assert data_name.startswith('output_'), 'only input_ and output_ prefixes are expected'
            outputs += 1
    
    print(f'{{"{test_name}", {inputs}, {outputs}}},')
print('}')
