for init_lr_theta in 0.001 0.005 0.01 0.05; do
  for init_lr_mu in 0.001 0.005 0.1 0.05; do
    for alpha in 0.001 0.005 0.01 0.05 ;do
      for init_mu in 1 5;do
        echo $init_lr_theta
        echo $init_lr_mu
        echo $alpha
        echo $init_mu
        # echo passward | sudo -S sh -c 'echo 1 >/proc/sys/vm/drop_caches' && sudo -S sh -c 'echo 2 >/proc/sys/vm/drop_caches' && sudo -S sh -c 'echo 3 >/proc/sys/vm/drop_caches';
        python main.py --init_lr_theta $init_lr_theta --init_lr_mu $init_lr_mu --alpha $alpha --alpha $alpha --init_mu $init_mu;
			done
		done
	done
done