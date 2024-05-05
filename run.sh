for init_lr_theta in 0.23; do
    for d_0 in 0.001 0.005;do
	   for init_lr_mu in 0.01 ; do 
		   for alpha in 1.0; do 
        		echo $init_lr_theta
        		echo $d_0
			echo $init_lr_mu
			echo $alpha
        		# echo passward | sudo -S sh -c 'echo 1 >/proc/sys/vm/drop_caches' && sudo -S sh -c 'echo 2 >/proc/sys/vm/drop_caches' && sudo -S sh -c 'echo 3 >/proc/sys/vm/drop_caches';
        		python main_biggrid.py --init_lr_theta $init_lr_theta --d_0 $d_0 --init_lr_mu $init_lr_mu --alpha $alpha;
			done
		done
	done
done


# 8000
