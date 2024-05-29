import os
root = '/mlainas/yewon/bp-benchmark/code/train/real_model'
join = os.path.join
model_fold = {"resnet1d":{
                "bcg": 
                        {0: 
                                {0: join(root, 'bcg-resnet1d-fold0-original-seed0.ckpt'),
                                1: join(root, 'bcg-resnet1d-fold1-original-seed0.ckpt'),
                                2: join(root, 'bcg-resnet1d-fold2-original-seed0.ckpt'),
                                3: join(root, 'bcg-resnet1d-fold3-original-seed0.ckpt'),
                                4: join(root, 'bcg-resnet1d-fold4-original-seed0.ckpt')},
                        1: 
                                {0: join(root, 'bcg-resnet1d-fold0-original-seed1.ckpt'),
                                1: join(root, 'bcg-resnet1d-fold1-original-seed1.ckpt'),
                                2: join(root, 'bcg-resnet1d-fold2-original-seed1.ckpt'),
                                3: join(root, 'bcg-resnet1d-fold3-original-seed1.ckpt'),
                                4: join(root, 'bcg-resnet1d-fold4-original-seed1.ckpt')},
                        2:
                                {0: join(root, 'bcg-resnet1d-fold0-original-seed2.ckpt'),
                                1: join(root, 'bcg-resnet1d-fold1-original-seed2.ckpt'),
                                2: join(root, 'bcg-resnet1d-fold2-original-seed2.ckpt'),
                                3: join(root, 'bcg-resnet1d-fold3-original-seed2.ckpt'),
                                4: join(root, 'bcg-resnet1d-fold4-original-seed2.ckpt')}
                        },
                "ppgbp": 
                        {0: 
                                {0: join(root, 'ppgbp-resnet1d-fold0-original-seed0.ckpt'),
                                1: join(root, 'ppgbp-resnet1d-fold1-original-seed0.ckpt'),
                                2: join(root, 'ppgbp-resnet1d-fold2-original-seed0.ckpt'),
                                3: join(root, 'ppgbp-resnet1d-fold3-original-seed0.ckpt'),
                                4: join(root, 'ppgbp-resnet1d-fold4-original-seed0.ckpt')},
                        1: 
                                {0: join(root, 'ppgbp-resnet1d-fold0-original-seed1.ckpt'),
                                1: join(root, 'ppgbp-resnet1d-fold1-original-seed1.ckpt'),
                                2: join(root, 'ppgbp-resnet1d-fold2-original-seed1.ckpt'),
                                3: join(root, 'ppgbp-resnet1d-fold3-original-seed1.ckpt'),
                                4: join(root, 'ppgbp-resnet1d-fold4-original-seed1.ckpt')},
                        2: 
                                {0: join(root, 'ppgbp-resnet1d-fold0-original-seed2.ckpt'),
                                1: join(root, 'ppgbp-resnet1d-fold1-original-seed2.ckpt'),
                                2: join(root, 'ppgbp-resnet1d-fold2-original-seed2.ckpt'),
                                3: join(root, 'ppgbp-resnet1d-fold3-original-seed2.ckpt'),
                                4: join(root, 'ppgbp-resnet1d-fold4-original-seed2.ckpt')}
                        },
                "sensors": 
                        {0: 
                                {0: join(root, 'sensors-resnet1d-fold0-original-seed0.ckpt'),
                                1: join(root, 'sensors-resnet1d-fold1-original-seed0.ckpt'),
                                2: join(root, 'sensors-resnet1d-fold2-original-seed0.ckpt'),
                                3: join(root, 'sensors-resnet1d-fold3-original-seed0.ckpt'),
                                4: join(root, 'sensors-resnet1d-fold4-original-seed0.ckpt')},
                        1: 
                                {0: join(root, 'sensors-resnet1d-fold0-original-seed1.ckpt'),
                                1: join(root, 'sensors-resnet1d-fold1-original-seed1.ckpt'),
                                2: join(root, 'sensors-resnet1d-fold2-original-seed1.ckpt'),
                                3: join(root, 'sensors-resnet1d-fold3-original-seed1.ckpt'),
                                4: join(root, 'sensors-resnet1d-fold4-original-seed1.ckpt')},
                        2: 
                                {0: join(root, 'sensors-resnet1d-fold0-original-seed2.ckpt'),
                                1: join(root, 'sensors-resnet1d-fold1-original-seed2.ckpt'),
                                2: join(root, 'sensors-resnet1d-fold2-original-seed2.ckpt'),
                                3: join(root, 'sensors-resnet1d-fold3-original-seed2.ckpt'),
                                4: join(root, 'sensors-resnet1d-fold4-original-seed2.ckpt')}
                        },
                "uci2": 
                        {0:
                                {0: join(root, 'uci2-resnet1d-fold0-original-seed0.ckpt')},
                        1:
                                {0: join(root, 'uci2-resnet1d-fold0-original-seed1.ckpt')},
                        2:
                                {0: join(root, 'uci2-resnet1d-fold0-original-seed2.ckpt')},        
                        }
                },
                "mlpbp":{
                "bcg": 
                        {0: 
                                {0: join(root, 'bcg-mlpbp-fold0-original-seed0.ckpt'),
                                1: join(root, 'bcg-mlpbp-fold1-original-seed0.ckpt'),
                                2: join(root, 'bcg-mlpbp-fold2-original-seed0.ckpt'),
                                3: join(root, 'bcg-mlpbp-fold3-original-seed0.ckpt'),
                                4: join(root, 'bcg-mlpbp-fold4-original-seed0.ckpt')},
                        },
                "ppgbp": 
                        {0: 
                                {0: join(root, 'ppgbp-mlpbp-fold0-original-seed0.ckpt'),
                                1: join(root, 'ppgbp-mlpbp-fold1-original-seed0.ckpt'),
                                2: join(root, 'ppgbp-mlpbp-fold2-original-seed0.ckpt'),
                                3: join(root, 'ppgbp-mlpbp-fold3-original-seed0.ckpt'),
                                4: join(root, 'ppgbp-mlpbp-fold4-original-seed0.ckpt')},
                        },
                "sensors": 
                        {0: 
                                {0: join(root, 'sensors-mlpbp-fold0-original-seed0.ckpt'),
                                1: join(root, 'sensors-mlpbp-fold1-original-seed0.ckpt'),
                                2: join(root, 'sensors-mlpbp-fold2-original-seed0.ckpt'),
                                3: join(root, 'sensors-mlpbp-fold3-original-seed0.ckpt'),
                                4: join(root, 'sensors-mlpbp-fold4-original-seed0.ckpt')},
                "uci2": 
                        {0:
                                {0: join(root, 'uci2-mlpbp-fold0-original-seed0.ckpt')},   
                                }
                        }
                }
        }