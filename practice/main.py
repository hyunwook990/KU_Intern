import os
# 사용할 모델의 파일 이름을 지정
from practice.XOR import train, test

if (__name__ == "__main__"):
    
    root_dir = "/gdrive/My_Drive/colab/ann/xor"
    output_dir = os.path.join(root_dir, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    input_data = "{0:s}/{1:s}".format(root_dir, "train_data.txt")
    
    config = {"mode": "train",
          # 특정 epoch마다 저장된 모델을 사용
          "model_name": "epoch_{0:d}.pt".format(1000),
          "output_dir": output_dir,
          "input_data": input_data,
          "input_node": 2,
          "hidden_node": 10,
          "output_node": 1,
          "learning_rate": 1,
          "batch_size": 4,
          "epoch": 1000,
}
    if config["mode"] == "train":
        train(config)
    else:
        test(config)
        
# from RNN(Many-to-One)_7 import train, test
# if (__name__ == "__main__"):
    
#     root_dir = "/gdrive/My_Drive/colab/rnn/stock"
#     output_dir = os.path.exsist(root_dir, "output")
#     if not os.path.join(output_dir):
#         os.makedirs(output_dir)
    
#     config = {"mode": "train",
#           # 특정 epoch마다 저장된 모델을 사용
#           "model_name": "epoch_{0:d}.pt".format(10),
#           "output_dir": output_dir,
#           "file_name": "{0:s}/samsung-2020.csv".format(root_dir),
#           "sequence_len": 3,
#           "input_size": 4,
#           "hidden_size": 10,
#           "output_size": 1,
#           "num_layers": 1,
#           "batch_size": 1,
#           "learning_rate": 0.1,
#           "epoch": 10,
# }
#     if config["mode"] == "train":
#         train(config)
#     else:
#         test(config)

# from Sentiment_Analysis import train, test
# if (__name__ == "__main__"):
    
#     output_dir = os.path.join(root_dir, "output")
#     cache_dir = os.path.join(root_dir, "cace")

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     if not os.path.exists(cache_dir):
#         os.makedirs(cache_dir) 
    
#     config = {"mode": "train",
#               "train_data_path": os.path. join(root_dir, "train_datas_wordpiece.txt"),
#               "test_data_path": os.path.join(root_dir, "test_datas_wordpiece.txt"),
#               "output_dir_path": output_dir,
#               "cache_dir_path": cache_dir,
#               "pretrained_nodel _name_or _path": "monologg/kobert",
#               "label_vocab_data_path": os.path.join(root_dir, "label_vocab.txt"),
#               "num_labels": 2,
#               "max_length": 142,
#               "epoch": 10,
#               "batch_size": 64
#               }

#     if config["mode"] == "train":
#         train(config)
#     else:
#         test(config)