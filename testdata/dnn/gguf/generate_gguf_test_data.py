import numpy as np
import gguf

if __name__ == '__main__':
    writer= gguf.gguf_writer.GGUFWriter("data/dummy.gguf", "dummy")
    writer.add_key_value("metadata1", "42", gguf.GGUFValueType.STRING)


    weight = np.ones((32,6), dtype=np.float32)
    bias = np.ones((32), dtype=np.float32)
    writer.add_tensor("blk.0.attn_qkv.weight", weight)
    writer.add_tensor("blk.0.attn_qkv.bias", bias)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()

    writer.close()