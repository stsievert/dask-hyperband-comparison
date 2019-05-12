install:
	sudo yum install -y tmux git htop
	pip install dask distributed --upgrade --user
	pip install git+https://github.com/stsievert/dask-ml@hyperband --user
	sudo /opt/conda/bin/conda install -y pytorch-cpu torchvision-cpu -c pytorch
	pip install skorch --upgrade --user

worker:
	# Run an c5.4xlarge
	# Scheduler on c5.2xlarge
	export OMP_NUM_THREADS=2; dask-worker --nprocs 5 --nthreads 1 --memory-limit="4GB" 172.31.20.153:8786

