for init_lr_theta in 1; do
  for init_lr_mu in 0.1; do
    for alpha in 0.1;do
      for init_mu in 1;do
        for max_step in 50 100 200 300 400; do
          for gamma in 0.8 0.85 0.9 0.95 0.99; do
            for d_0 in 0.01 0.1 1; do
              echo $gamma
              echo $max_step
              echo $init_lr_theta
              echo $init_lr_mu
              echo $alpha
              echo $init_mu
              echo $d_0
              # echo passward | sudo -S sh -c 'echo 1 >/proc/sys/vm/drop_caches' && sudo -S sh -c 'echo 2 >/proc/sys/vm/drop_caches' && sudo -S sh -c 'echo 3 >/proc/sys/vm/drop_caches';
              python main_mc.py --d_0 $d_0 --gamma $gamma --max_step $max_step --init_lr_theta $init_lr_theta --init_lr_mu $init_lr_mu --alpha $alpha --alpha $alpha --init_mu $init_mu;
            done
          done
        done
			done
		done
	done
done