for init_lr_theta in 0.24; do
    for d_0 in 0.001 0.005 0.01 0.05 ;do
        echo $init_lr_theta
        echo $d_0
        # echo passward | sudo -S sh -c 'echo 1 >/proc/sys/vm/drop_caches' && sudo -S sh -c 'echo 2 >/proc/sys/vm/drop_caches' && sudo -S sh -c 'echo 3 >/proc/sys/vm/drop_caches';
        python main.py --init_lr_theta $init_lr_theta --d_0 $d_0 ;
		done
done


# 8000