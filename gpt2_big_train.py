import gpt_2_simple as gpt2
import tensorflow as tf
file_name = "all_ggw.txt"
sess = gpt2.start_tf_sess()
#reuse=tf.AUTO_REUSE
gpt2.finetune(sess,
              dataset=file_name,
              model_name='1558M',
              steps=30000,
              run_name='after_all_ggw_final',
              print_every=10,
              sample_every=20000000,
              save_every=50, 
              multi_gpu=True, 
              batch_size=14, #16 works best till now
              overwrite=True, 
              restore_from='checkpoint/after_all_ggw_final'
              )