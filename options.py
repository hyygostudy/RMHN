import argparse

def arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--clamp",default=2.0,type=float,help='a constant factor',required=False
    )
    parser.add_argument(
        "--device_ids", default=[0], type=int, help='a constant factor', required=False
    )
    parser.add_argument(
        "--init_scale",default=0.01,type=float,help=" init_weight_scale for model initial",required=False
    )
    parser.add_argument(
        '--lr',default=10 ** -4.5,type=float,help='setting learning rate',required=False
    )
    parser.add_argument(
        '--betas',default=(0.5, 0.999),type=tuple,help='betas setting for Adam',required=False
    )
    parser.add_argument(
        '--eps',default=1e-6,type=float,help='prevent zero',required=False
    )
    parser.add_argument(
        '--weight_decay',default=1e-5,type=float,help='weight decay',required=False
    )
    parser.add_argument(
        '--weight_step',default=1000,type=int,help='Learning rate decay interval',required=False
    )
    parser.add_argument(
        '--gamma',default=0.5,type=float,help='current lr = original lr*gamma',required=False
    )
    parser.add_argument(
        '--channels_in',default=3,type=int,help='the input channels',required=False
    )
    parser.add_argument(
        '--lamda_reconstruction',default=3,type=int,help='the weight for balancing secret-rev',required=False
    )
    parser.add_argument(
        '--lamda_guide',default=1,type=int,help='the weight for balancing stego',required=False
    )
    parser.add_argument(
        '--lamda_low_F',default=1,type=int,help='the weight for balaning stego in low frequency',required=False
    )
    parser.add_argument(
        '--train_image_dir',default=' ',type=str,help='the image path for training',required=False
    )
    parser.add_argument(
        '--image_patch_size',default=512,type=int,help='training patch size',required=False
    )
    parser.add_argument(
        '--training_batch',default=4,type=int,help='the training batchsize',required=False
    )
    parser.add_argument(
        '--epoch',default=20000,type=int,help='the total training epoch',required=False
    )
    parser.add_argument(
        '--save_frequency',default=1000,type=int,help=' save model per ? epochs',required=False
    )
    parser.add_argument(
        '--model_path1',default='E:/pythonProjects/RMSN/model/model1.pt',help='save your model to this path',required=False
    )
    parser.add_argument(
        '--model_path2',default='E:/pythonProjects/RMSN/model/model2.pt',help='save your model to this path',required=False
    )
    parser.add_argument(
        '--model_path3',default='E:/pythonProjects/RMSN/model/model3.pt',help='save your model to this path',required=False
    )
    parser.add_argument(
        '--val_frequency',default=100,type=int,help='validate your model per ? epochs',required=False
    )
    parser.add_argument(
        '--testpath',default='E:/datasets/test/',type=str,help='test image path',required=False
    )
    parser.add_argument(
        '--cropsize_test',default=1024,type=int,help=' the size of the test image',required=False
    )
    parser.add_argument(
        '--test_cover_path',default='E:/pythonProjects/RMSN/image/cover/'
    )
    parser.add_argument(
        '--test_secret_path1',default='E:/pythonProjects/RMSN/image/secret1/'
    )
    parser.add_argument(
        '--test_secret_path2',default='E:/pythonProjects/RMSN/image/secret2/'
    )
    parser.add_argument(
        '--test_secret_path3',default='E:/pythonProjects/RMSN/image/secret3/'
    )
    parser.add_argument(
        '--test_stego_path1',default='E:/pythonProjects/RMSN/image/stego1/'
    )
    parser.add_argument(
        '--test_stego_path2',default='E:/pythonProjects/RMSN/image/stego2/'
    )
    parser.add_argument(
        '--test_stego_path3',default='E:/pythonProjects/RMSN/image/stego3/'
    )
    parser.add_argument(
        '--test_recs_path1',default='E:/pythonProjects/RMSN/image/recs1/'
    )
    parser.add_argument(
        '--test_recs_path2',default='E:/pythonProjects/RMSN/image/recs2/'
    )
    parser.add_argument(
        '--test_recs_path3',default='E:/pythonProjects/RMSN/image/recs3/'
    )
    parser.add_argument(
        '--test_cwrong_map_path1',default='E:/pythonProjects/RMSN/image/cwrong map1/'
    )
    parser.add_argument(
        '--test_cwrong_map_path2',default='E:/pythonProjects/RMSN/image/cwrong map2/'
    )
    parser.add_argument(
        '--test_cwrong_map_path3',default='E:/pythonProjects/RMSN/image/cwrong map3/'
    )
    parser.add_argument(
        '--test_swrong_map_path1',default='E:/pythonProjects/RMSN/image/swrong map1/'
    )
    parser.add_argument(
        '--test_swrong_map_path2',default='E:/pythonProjects/RMSN/image/swrong map2/'
    )
    parser.add_argument(
        '--test_swrong_map_path3',default='E:/pythonProjects/RMSN/image/swrong map3/'
    )
    parser.add_argument(
        '--x_LL',default='E:/pythonProjects/RMSN/image/x_LL/'
    )
    parser.add_argument(
        '--x_HL',default='E:/pythonProjects/RMSN/image/x_HL/'
    )
    parser.add_argument(
        '--x_LH',default='E:/pythonProjects/RMSN/image/x_LH/'
    )
    parser.add_argument(
        '--x_HH',default='E:/pythonProjects/RMSN/image/x_HH/'
    )
    opt=parser.parse_args()
    return opt